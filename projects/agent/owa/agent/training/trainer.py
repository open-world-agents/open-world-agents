import os
from typing import Optional

from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction

# from transformers.utils import add_start_docstrings
from trl import SFTConfig, SFTTrainer


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
    # TODO: implement fine-grained metrics considering event decoding.
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
        ignore_keys.append("image_hidden_states")
        # Run standard evaluation
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        # Save prediction examples on main process
        if self.is_world_process_zero() and self.args.eval_samples_to_show > 0:
            self._save_predictions_and_ground_truth(output)

        return output

    def _save_predictions_and_ground_truth(self, output: EvalLoopOutput):
        """Save predictions and ground truth with per-sample metrics."""
        assert not isinstance(output.predictions, tuple)
        logits = output.predictions  # Raw token logits (model predictions)
        labels = output.label_ids  # Raw token labels (ground truth)
        losses = getattr(output, "losses", None)

        if logits is None or labels is None or self.processing_class is None:
            return

        # Process each sample to calculate metrics and decode text
        data = []
        for i in range(len(logits)):
            label = labels[i]

            # === Copied from SFTTrainer compute_loss
            shift_logits = logits[i][..., :-1, :]
            shift_labels = label[..., 1:]

            # Get predictions
            predictions = shift_logits.argmax(axis=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            # correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            # total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0

            # === End copy

            # Extract per-sample loss
            sample_loss = losses[i].item() if losses is not None else None

            # Decode tokens to text. NOTE: `mask` is needed because TokenizerFast can't decode -100. Refer to https://github.com/huggingface/transformers/issues/31110#issuecomment-2137712416
            pred_text = self.processing_class.decode(predictions[mask], skip_special_tokens=False).strip()
            label_text = self.processing_class.decode(shift_labels[mask], skip_special_tokens=False).strip()

            data.append(
                {
                    "id": i,
                    "prediction": pred_text,
                    "ground_truth": label_text,
                    "pred_tokens": predictions.tolist(),
                    "gt_tokens": label.tolist(),
                    "token_accuracy": round(accuracy, 3),
                    "loss": round(sample_loss, 4) if sample_loss is not None else None,
                }
            )

        # Calculate summary statistics
        total = len(data)
        avg_token_acc = sum(d["token_accuracy"] for d in data) / total
        avg_loss = (
            sum(d["loss"] for d in data if d["loss"] is not None) / total
            if any(d["loss"] is not None for d in data)
            else None
        )

        # Create output directory and save results
        step = self.state.global_step
        output_dir = os.path.join(self.args.output_dir or "./output", "eval")
        os.makedirs(output_dir, exist_ok=True)

        # Write evaluation report to markdown file
        with open(os.path.join(output_dir, f"eval_step_{step}.md"), "w") as f:
            f.write(f"# Step {step}\n\n")
            f.write(f"**Token Accuracy:** {avg_token_acc:.1%}\n")
            if avg_loss is not None:
                f.write(f"**Average Loss:** {avg_loss:.3f}\n")
            f.write("\n")

            # Write sample predictions and ground truth
            for d in data[: self.args.eval_samples_to_show]:
                loss_str = f" | Loss: {d['loss']:.3f}" if d["loss"] is not None else ""
                f.write(f"## {d['id']} | Acc: {d['token_accuracy']:.1%}{loss_str}\n")
                f.write(f"**Pred:** {d['prediction']}\n")
                f.write(f"**True:** {d['ground_truth']}\n")
                f.write(f"**Pred Tokens:** {d['pred_tokens']}\n")
                f.write(f"**GT Tokens:** {d['gt_tokens']}\n\n")
