import os
from typing import Optional, cast

import torch
from torch.utils.data import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction

# from transformers.utils import add_start_docstrings
from trl import SFTConfig, SFTTrainer

from .event_metric import compute_metrics_for_events


# @add_start_docstrings(SFTConfig.__doc__)
class OWASFTConfig(SFTConfig):
    batch_eval_metrics: bool = True  # False in SFTConfig
    prediction_loss_only: bool = False  # True in SFTConfig
    eval_samples_to_show: int = 16  # Number of samples to show in eval output


def compute_metrics(eval_pred: EvalPrediction, compute_result: bool = False) -> dict:
    """Placeholder metrics function for evaluation.
    Args:
        eval_pred (EvalPrediction): Evaluation predictions
        compute_result (bool): Whether to compute metrics. Given to True for last evaluation batch.
    """
    del eval_pred  # Unused parameter
    return {}


class OWASFTTrainer(SFTTrainer):
    """Custom SFT Trainer that saves predicted sequences and ground truth during evaluation."""

    args: OWASFTConfig

    def __init__(self, args: OWASFTConfig, compute_metrics=compute_metrics, **kwargs):
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

        # Compute and add event-based metrics to output
        event_metrics = self._compute_event_metrics(output, metric_key_prefix)
        if event_metrics:
            # Create new metrics dict with existing metrics plus event metrics
            existing_metrics = output.metrics or {}
            updated_metrics = {**existing_metrics, **event_metrics}
            # Create new EvalLoopOutput with updated metrics
            output = EvalLoopOutput(
                predictions=output.predictions,
                label_ids=output.label_ids,
                metrics=updated_metrics,
                num_samples=output.num_samples,
            )

        # Save prediction examples on main process
        if self.is_world_process_zero() and self.args.eval_samples_to_show > 0:
            self._save_predictions_and_ground_truth(output)

        return output

    def _compute_event_metrics(self, output: EvalLoopOutput, metric_key_prefix: str = "eval") -> dict:
        """Compute event-based metrics from evaluation output."""
        if output.predictions is None or output.label_ids is None:
            return {}

        assert not isinstance(output.predictions, tuple)
        logits = output.predictions
        labels = output.label_ids

        # Get the tokenizer from processing_class
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        if not hasattr(tokenizer, "decode"):
            tokenizer = self.processing_class if hasattr(self.processing_class, "decode") else None

        if tokenizer is None or not hasattr(tokenizer, "decode"):
            return {}

        # Cast to PreTrainedTokenizer for type checking
        tokenizer = cast(PreTrainedTokenizer, tokenizer)

        # Compute event metrics for all samples
        all_sample_metrics = []
        for i in range(len(logits)):
            label = labels[i]
            shift_logits = logits[i][..., :-1, :]
            shift_labels = label[..., 1:]

            try:
                sample_metrics = compute_metrics_for_events(
                    logits=shift_logits,
                    labels=shift_labels,
                    tokenizer=tokenizer,
                )
                all_sample_metrics.append(sample_metrics)
            except Exception:
                # Skip samples that fail event metric computation
                continue

        # Aggregate metrics across all samples
        aggregated_metrics = self._aggregate_sample_metrics(all_sample_metrics)

        # Add metric prefix to all keys
        prefixed_metrics = {}
        for key, value in aggregated_metrics.items():
            prefixed_key = f"{metric_key_prefix}_{key}"
            prefixed_metrics[prefixed_key] = value

        return prefixed_metrics

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

    def _aggregate_sample_metrics(self, all_sample_metrics: list) -> dict:
        """Aggregate event metrics across all samples."""
        if not all_sample_metrics:
            return {}

        # Collect all unique metric keys from all samples
        all_metric_keys = set()
        for metrics in all_sample_metrics:
            all_metric_keys.update(metrics.keys())

        # Remove error keys from aggregation
        all_metric_keys.discard("error")

        aggregated = {}
        for key in sorted(all_metric_keys):
            # Collect values for this key from all samples that have it
            values = [
                metrics[key]
                for metrics in all_sample_metrics
                if key in metrics and isinstance(metrics[key], (int, float))
            ]

            if not values:  # Skip if no numeric values found
                continue

            # Use simple averaging for all metrics
            aggregated[key] = sum(values) / len(values)

        return aggregated
