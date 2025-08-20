import os
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction

# from transformers.utils import add_start_docstrings
from trl import SFTConfig, SFTTrainer

from owa.data.collator import ModelType, detect_model_type
from owa.data.episode_tokenizer import EpisodeTokenizer, EpisodeTokenizerConfig


# @add_start_docstrings(SFTConfig.__doc__)
@dataclass
class OWASFTConfig(SFTConfig):
    batch_eval_metrics: bool = True  # False in SFTConfig
    prediction_loss_only: bool = False  # True in SFTConfig
    eval_samples_to_show: int = 16  # Number of samples to show in eval output


class OWAEvaluatorBatched:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.metrics = {
            "comparable_events": 0,
            "total_events": 0,
            "keyboard_accuracy_correct": 0,
            "keyboard_accuracy_total": 0,
            "mouse_movement_squared_error": 0,
            "mouse_movement_gt_values": [],
            "mouse_movement_percentage_errors": [],
            "mouse_action_correct": 0,
            "mouse_action_total": 0,
            "mouse_scroll_correct": 0,
            "mouse_scroll_total": 0,
        }

        model_type = detect_model_type(tokenizer.name_or_path)
        # TODO: make these configurable, load and save
        if model_type == ModelType.INTERNVL:
            # InternVL3 configuration
            episode_tokenizer_config = EpisodeTokenizerConfig(
                encoder_type="hierarchical",
                fake_image_placeholder="<fake_image_placeholder>",
                image_token_prefix="<img>",
                image_token="<IMG_CONTEXT>",
                image_token_length=256,
                image_token_suffix="</img>",
                episode_start_token="<EPISODE_START>",
                episode_end_token="<EPISODE_END>",
            )
        else:
            # SmolVLM2 and other models configuration
            episode_tokenizer_config = EpisodeTokenizerConfig(
                encoder_type="hierarchical",
                fake_image_placeholder="<fake_image_placeholder>",
                image_token_prefix="<fake_token_around_image><global-img>",
                image_token="<image>",
                image_token_length=64,
                image_token_suffix="<fake_token_around_image>",
                episode_start_token="<EPISODE_START>",
                episode_end_token="<EPISODE_END>",
            )
        self.episode_tokenizer = EpisodeTokenizer(episode_tokenizer_config)
        self.episode_tokenizer.prepare_model(tokenizer=tokenizer)

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = False) -> dict:
        """Placeholder metrics function for evaluation.
        Args:
            eval_pred (EvalPrediction): Evaluation predictions
            compute_result (bool): Whether to compute metrics. Given to True for last evaluation batch.
        """
        logits = eval_pred.predictions  # [batch_size, seq_len, vocab_size]
        labels = eval_pred.label_ids  # [batch_size, seq_len]
        # WHY THE HELL THESE ARE NOT NUMPY ARRAYS??? TYPE HINT LIEING
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = cast(torch.Tensor, logits).cpu().numpy()

        if isinstance(labels, tuple):
            labels = labels[0]
        labels = cast(torch.Tensor, labels).cpu().numpy()
        assert isinstance(logits, np.ndarray) and isinstance(labels, np.ndarray), (
            f"Expected numpy arrays for logits and labels, got {type(logits)} and {type(labels)}"
        )

        # Shift logits and labels
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        for logits, labels in zip(shift_logits, shift_labels):
            predictions = logits.argmax(axis=-1)

            # Since logits/labels is shifted-by-1, if the first token is not <EVENT_START>, add it.
            event_start_token_id = self.tokenizer.convert_tokens_to_ids("<EVENT_START>")
            event_end_token_id = self.tokenizer.convert_tokens_to_ids("<EVENT_END>")
            if labels[0] != event_start_token_id:
                predictions = np.insert(predictions, 0, event_start_token_id)
                labels = np.insert(labels, 0, event_start_token_id)

            # Extract event boundaries
            where_start = np.where(labels == event_start_token_id)[0]

            for start in where_start:
                # find the place where first event_end token appears after start
                end = np.where(labels[start:] == event_end_token_id)[0][0] + start + 1

                # Mask out padding and image tokens
                mask = labels[start:end] != -100
                event_prediction = predictions[start:end][mask]
                event_label = labels[start:end][mask]

                # Decode tokens to events with episode tokenizer
                pred_events = list(self.episode_tokenizer.decode_episode(event_prediction, skip_invalid=True))
                gt_events = list(self.episode_tokenizer.decode_episode(event_label, skip_invalid=True))

                # Count total events
                self.metrics["total_events"] += 1

                if not (len(pred_events) == len(gt_events) == 1):
                    continue

                pred_event = pred_events[0]
                gt_event = gt_events[0]

                if pred_event.topic != gt_event.topic:
                    continue

                # Count comparable events
                self.metrics["comparable_events"] += 1

                # for keyboard event, compute accuracy for both vk and event_type being correct
                if pred_event.topic == "keyboard" and gt_event.topic == "keyboard":
                    pred_decoded = pred_event.decoded
                    gt_decoded = gt_event.decoded
                    # Only count as correct if both vk and event_type are correct
                    both_correct = (pred_decoded.vk == gt_decoded.vk) and (
                        pred_decoded.event_type == gt_decoded.event_type
                    )
                    self.metrics["keyboard_accuracy_correct"] += int(both_correct)
                    self.metrics["keyboard_accuracy_total"] += 1
                # for mouse event, compute R^2 and percentage error for (dx, dy) vector
                if pred_event.topic == "mouse/raw" and gt_event.topic == "mouse/raw":
                    pred_decoded = pred_event.decoded
                    gt_decoded = gt_event.decoded
                    # Compute squared Euclidean distance between predicted and ground truth vectors
                    dx_error = pred_decoded.dx - gt_decoded.dx
                    dy_error = pred_decoded.dy - gt_decoded.dy
                    self.metrics["mouse_movement_squared_error"] += dx_error**2 + dy_error**2
                    self.metrics["mouse_movement_gt_values"].append((gt_decoded.dx, gt_decoded.dy))

                    # Compute percentage error using Euclidean distance between vectors
                    euclidean_error = np.sqrt(
                        (pred_decoded.dx - gt_decoded.dx) ** 2 + (pred_decoded.dy - gt_decoded.dy) ** 2
                    )
                    gt_norm = np.sqrt(gt_decoded.dx**2 + gt_decoded.dy**2)
                    if gt_norm > 0:  # Avoid division by zero
                        percentage_error = euclidean_error / gt_norm * 100
                        self.metrics["mouse_movement_percentage_errors"].append(percentage_error)
                # for mouse event which has button operation, compute accuracy for button_flags and button_data
                if pred_event.topic == "mouse/raw" and gt_event.topic == "mouse/raw":
                    pred_decoded = pred_event.decoded
                    gt_decoded = gt_event.decoded
                    if gt_decoded.button_flags != 0:
                        self.metrics["mouse_action_correct"] += int(
                            pred_decoded.button_flags == gt_decoded.button_flags
                        )
                        self.metrics["mouse_action_total"] += 1
                    if gt_decoded.button_data != 0:
                        self.metrics["mouse_scroll_correct"] += int(pred_decoded.button_data == gt_decoded.button_data)
                        self.metrics["mouse_scroll_total"] += 1

        # Return metrics only for the last batch
        if compute_result:
            final_metrics = {}

            # Calculate comparable events ratio
            if self.metrics["total_events"] > 0:
                final_metrics["comparable"] = self.metrics["comparable_events"] / self.metrics["total_events"]

            # Calculate accuracy metrics
            if self.metrics["keyboard_accuracy_total"] > 0:
                final_metrics["keyboard_accuracy"] = (
                    self.metrics["keyboard_accuracy_correct"] / self.metrics["keyboard_accuracy_total"]
                )
            if self.metrics["mouse_action_total"] > 0:
                final_metrics["mouse_action_accuracy"] = (
                    self.metrics["mouse_action_correct"] / self.metrics["mouse_action_total"]
                )
            if self.metrics["mouse_scroll_total"] > 0:
                final_metrics["mouse_scroll_accuracy"] = (
                    self.metrics["mouse_scroll_correct"] / self.metrics["mouse_scroll_total"]
                )

            # Calculate R² and percentile metrics for mouse movement (2D vector)
            if len(self.metrics["mouse_movement_gt_values"]) > 0:
                gt_vectors = np.array(self.metrics["mouse_movement_gt_values"])  # Shape: (n_samples, 2)

                # R² = 1 - (SS_res / SS_tot)
                # SS_res = sum of squared residuals (already computed as Euclidean distances)
                # SS_tot = sum of squared deviations from mean vector
                mean_vector = np.mean(gt_vectors, axis=0)  # Shape: (2,)

                # Compute total sum of squares for the 2D vector
                ss_tot = np.sum(np.sum((gt_vectors - mean_vector) ** 2, axis=1))

                if ss_tot > 0:
                    final_metrics["mouse_movement_r2"] = 1 - (self.metrics["mouse_movement_squared_error"] / ss_tot)
                else:
                    final_metrics["mouse_movement_r2"] = 1.0  # Perfect prediction if no variance

                # Calculate percentile-based percentage error metrics (p0, p25, p50, p75, p100)
                if len(self.metrics["mouse_movement_percentage_errors"]) > 0:
                    pe_values = np.array(self.metrics["mouse_movement_percentage_errors"])
                    final_metrics["mouse_movement_pe_p0"] = np.percentile(pe_values, 0)  # min
                    final_metrics["mouse_movement_pe_p25"] = np.percentile(pe_values, 25)  # 1st quartile
                    final_metrics["mouse_movement_pe_p50"] = np.percentile(pe_values, 50)  # median
                    final_metrics["mouse_movement_pe_p75"] = np.percentile(pe_values, 75)  # 3rd quartile
                    final_metrics["mouse_movement_pe_p100"] = np.percentile(pe_values, 100)  # max

            # Reset metrics for next evaluation
            self.metrics = {
                "comparable_events": 0,
                "total_events": 0,
                "keyboard_accuracy_correct": 0,
                "keyboard_accuracy_total": 0,
                "mouse_movement_squared_error": 0,
                "mouse_movement_gt_values": [],
                "mouse_movement_percentage_errors": [],
                "mouse_action_correct": 0,
                "mouse_action_total": 0,
                "mouse_scroll_correct": 0,
                "mouse_scroll_total": 0,
            }

            return final_metrics

        return {}


class OWASFTTrainer(SFTTrainer):
    """Custom SFT Trainer that saves predicted sequences and ground truth during evaluation."""

    args: OWASFTConfig

    def __init__(self, args: OWASFTConfig, **kwargs):
        tokenizer = cast(PreTrainedTokenizer, kwargs["processing_class"].tokenizer)
        compute_metrics = OWAEvaluatorBatched(tokenizer=tokenizer)
        kwargs = kwargs | dict(args=args, compute_metrics=compute_metrics)
        super().__init__(**kwargs)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        ignore_keys = ignore_keys or []
        ignore_keys += ["last_hidden_state", "past_key_values", "hidden_states", "attentions", "image_hidden_states"]
        # Run standard evaluation
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        # Save prediction examples on main process
        if self.is_world_process_zero() and self.args.eval_samples_to_show > 0:
            self._save_predictions_and_ground_truth(output)

        return output

    def _save_predictions_and_ground_truth(self, output: EvalLoopOutput):
        """Save predictions and ground truth examples for inspection."""
        assert not isinstance(output.predictions, tuple)
        logits = output.predictions  # Raw token logits (model predictions)
        labels = output.label_ids  # Raw token labels (ground truth)
        losses = getattr(output, "losses", None)

        if logits is None or labels is None or self.processing_class is None:
            return

        # Get the tokenizer from processing_class
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        if not hasattr(tokenizer, "decode"):
            # Fallback to processing_class if it has decode method
            tokenizer = self.processing_class if hasattr(self.processing_class, "decode") else None

        if tokenizer is None or not hasattr(tokenizer, "decode"):
            return

        # Cast to PreTrainedTokenizer for type checking
        tokenizer = cast(PreTrainedTokenizer, tokenizer)

        # Process each sample for text decoding and basic info
        data = []

        for i in range(len(logits)):
            label = labels[i]

            # ===== Prepare data for text decoding =====
            shift_logits = logits[i][..., :-1, :]
            shift_labels = label[..., 1:]

            # Get predictions for text decoding
            predictions = shift_logits.argmax(axis=-1)
            mask = shift_labels != -100

            # Calculate basic token accuracy for display
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0

            # Extract per-sample loss
            sample_loss = losses[i].item() if losses is not None else None

            # Decode tokens to text. NOTE: `mask` is needed because TokenizerFast can't decode -100. Refer to https://github.com/huggingface/transformers/issues/31110#issuecomment-2137712416
            pred_text = tokenizer.decode(predictions[mask], skip_special_tokens=False).strip()
            label_text = tokenizer.decode(shift_labels[mask], skip_special_tokens=False).strip()

            data.append(
                {
                    "id": i,
                    "prediction": pred_text,
                    "ground_truth": label_text,
                    "pred_tokens": predictions.tolist(),
                    "gt_tokens": shift_labels.tolist(),
                    "token_accuracy": round(accuracy, 3),
                    "loss": round(sample_loss, 4) if sample_loss is not None else None,
                }
            )

        # Extract aggregated metrics from the output.metrics (computed by _compute_event_metrics)
        aggregated_event_metrics = {}
        if output.metrics:
            # Remove the eval_ prefix for cleaner display in reports
            for key, value in output.metrics.items():
                if key.startswith("eval_"):
                    display_key = key[5:]  # Remove "eval_" prefix
                    aggregated_event_metrics[display_key] = value

        # Calculate summary statistics from data
        total = len(data)
        avg_token_acc = sum(d["token_accuracy"] for d in data) / total if total > 0 else 0.0
        avg_loss = (
            sum(d["loss"] for d in data if d["loss"] is not None) / total
            if any(d["loss"] is not None for d in data)
            else None
        )

        # Save evaluation results
        step = self.state.global_step
        output_dir = os.path.join(self.args.output_dir or "./output", "eval")
        os.makedirs(output_dir, exist_ok=True)

        # NOTE: since saved output is too large I save only first sample. e.g. for 256 sample output is 256*1024*50257*4 = 52GB
        # Save complete evaluation output. Without pickle argument `OverflowError: serializing a string larger than 4 GiB requires pickle protocol 4 or higher` raised
        torch.save(
            {"logits": logits[:1], "labels": labels[:1]},
            os.path.join(output_dir, f"eval_step_{step}.pt"),
            pickle_protocol=4,
        )

        # Save markdown with event metrics
        with open(os.path.join(output_dir, f"eval_step_{step}.md"), "w") as f:
            f.write(f"# Step {step}\n\n")
            f.write(f"**Token Accuracy (Basic):** {avg_token_acc:.1%}\n")
            if avg_loss is not None:
                f.write(f"**Average Loss:** {avg_loss:.3f}\n")

            # Write aggregated event metrics
            if aggregated_event_metrics:
                f.write("\n## Event-Based Metrics\n")
                for key, value in sorted(aggregated_event_metrics.items()):
                    if isinstance(value, float):
                        if key.endswith("_accuracy"):
                            f.write(f"**{key}:** {value:.1%}\n")
                        elif key.endswith("_loss"):
                            f.write(f"**{key}:** {value:.3f}\n")
                        else:
                            f.write(f"**{key}:** {value:.3f}\n")
                    else:
                        f.write(f"**{key}:** {value}\n")

            f.write("\n## Sample Predictions\n")

            for d in data[: self.args.eval_samples_to_show]:
                loss_str = f" | Loss: {d['loss']:.3f}" if d["loss"] is not None else ""
                f.write(f"### {d['id']} | Acc: {d['token_accuracy']:.1%}{loss_str}\n")
                f.write(f"**Pred:** {d['prediction']}\n**True:** {d['ground_truth']}\n")
                f.write(f"**Pred Tokens:** {d['pred_tokens']}\n**GT Tokens:** {d['gt_tokens']}\n\n")


if __name__ == "__main__":
    import dill as pickle
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/mnt/harbor/projects/owa/checkpoints/smolvlm-256m_vpt_0811-00")

    outputs = torch.load(
        "/mnt/harbor/projects/owa/checkpoints/smolvlm-256m_vpt_0811-00/eval/eval_step_12348.pt", pickle_module=pickle
    )
    logits = torch.from_numpy(outputs["logits"])
    labels = torch.from_numpy(outputs["labels"])
    eval_prediction = EvalPrediction(predictions=logits, label_ids=labels)

    evaluator = OWAEvaluatorBatched(tokenizer=tokenizer)
    metrics = evaluator(eval_prediction, compute_result=True)
    print(metrics)
