import json
import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Optional, TextIO, cast

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrainerState
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction

# from transformers.utils import add_start_docstrings
from trl import SFTConfig, SFTTrainer

from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.data.collator import ModelType, detect_model_type
from owa.data.episode_tokenizer import EpisodeTokenizer, EpisodeTokenizerConfig


def interquartile_mean(data: np.ndarray) -> float:
    """Calculate the Interquartile Mean (IQM) of the data.

    IQM is the mean of the values between the 25th and 75th percentiles.
    """
    if len(data) == 0:
        return 0.0

    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)

    # Filter data to include only values between Q1 and Q3 (inclusive)
    iqr_data = data[(data >= q25) & (data <= q75)]

    return float(np.mean(iqr_data)) if len(iqr_data) > 0 else 0.0


def stratified_bootstrap_ci(
    data: np.ndarray, n_bootstrap: int = 1000, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Calculate stratified bootstrap confidence intervals for IQM.

    Args:
        data: Input data array
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    if len(data) == 0:
        return (0.0, 0.0)

    # Generate bootstrap samples
    bootstrap_iqms = []
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility

    for _ in range(n_bootstrap):
        # Stratified sampling: sample with replacement
        bootstrap_sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_iqms.append(interquartile_mean(bootstrap_sample))

    bootstrap_iqms = np.array(bootstrap_iqms)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_iqms, lower_percentile)
    upper_bound = np.percentile(bootstrap_iqms, upper_percentile)

    return (float(lower_bound), float(upper_bound))


def movement_direction_error_degrees(pred_dx: float, pred_dy: float, gt_dx: float, gt_dy: float) -> float | None:
    """Calculate angular error between predicted and ground truth movement vectors.

    Args:
        pred_dx, pred_dy: Predicted movement vector components
        gt_dx, gt_dy: Ground truth movement vector components

    Returns:
        Angular error in degrees (0-180°), or None for zero movements
    """
    # Skip zero movements (no meaningful direction)
    if gt_dx == 0 and gt_dy == 0:
        return None

    # Calculate angles using atan2 (handles all quadrants correctly)
    pred_angle = np.arctan2(pred_dy, pred_dx)
    gt_angle = np.arctan2(gt_dy, gt_dx)

    # Calculate shortest angular distance (0 to π radians)
    angle_diff = np.abs(pred_angle - gt_angle)
    shortest_angle = np.minimum(angle_diff, 2 * np.pi - angle_diff)

    # Convert to degrees
    return float(shortest_angle * 180 / np.pi)


# @add_start_docstrings(SFTConfig.__doc__)
@dataclass
class OWASFTConfig(SFTConfig):
    batch_eval_metrics: bool = True  # False in SFTConfig
    prediction_loss_only: bool = False  # True in SFTConfig
    eval_samples_to_show: int = 16  # Number of samples to show in eval output


class OWAEvaluatorBatched:
    state: TrainerState

    def __init__(self, *, tokenizer: PreTrainedTokenizer, output_dir, eval_samples_to_show: int = 16):
        self.tokenizer = tokenizer
        self.eval_samples_to_show = eval_samples_to_show
        self.output_dir = output_dir
        self._initialize_metrics()

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

        self.visualize_queue = []

    def _initialize_metrics(self) -> None:
        """Initialize or reset all metrics to their default values."""
        self.metrics = {
            "comparable_events": 0,
            "total_events": 0,
            "keyboard_accuracy_correct": 0,
            "keyboard_accuracy_total": 0,
            "mouse_move_squared_error": 0,
            "mouse_move_gt_values": [],
            "mouse_move_pe_euclidean": [],
            "mouse_move_signed_pe_x": [],
            "mouse_move_signed_pe_y": [],
            "mouse_move_direction_errors": [],
            "mouse_move_accuracy_1000_correct": 0,
            "mouse_move_accuracy_1000_total": 0,
            "mouse_move_accuracy_100_correct": 0,
            "mouse_move_accuracy_100_total": 0,
            "mouse_move_accuracy_10_correct": 0,
            "mouse_move_accuracy_10_total": 0,
            "mouse_move_accuracy_1_correct": 0,
            "mouse_move_accuracy_1_total": 0,
            "timestamp_squared_error": 0,
            "timestamp_gt_values": [],
            "timestamp_pred_values": [],
            "timestamp_absolute_errors": [],
            "timestamp_signed_errors": [],
            "mouse_action_correct": 0,
            "mouse_action_total": 0,
            "mouse_scroll_correct": 0,
            "mouse_scroll_total": 0,
            # Per-event type comparable metrics
            "keyboard_comparable": 0,
            "mouse_op_comparable": 0,
            "mouse_nop_comparable": 0,
            "screen_comparable": 0,
            # Per-event type total counts
            "keyboard_total": 0,
            "mouse_op_total": 0,
            "mouse_nop_total": 0,
            "screen_total": 0,
            # Per-event type timestamp metrics
            "keyboard_timestamp_squared_error": 0,
            "keyboard_timestamp_gt_values": [],
            "keyboard_timestamp_absolute_errors": [],
            "keyboard_timestamp_signed_errors": [],
            "mouse_op_timestamp_squared_error": 0,
            "mouse_op_timestamp_gt_values": [],
            "mouse_op_timestamp_absolute_errors": [],
            "mouse_op_timestamp_signed_errors": [],
            "mouse_nop_timestamp_squared_error": 0,
            "mouse_nop_timestamp_gt_values": [],
            "mouse_nop_timestamp_absolute_errors": [],
            "mouse_nop_timestamp_signed_errors": [],
            "screen_timestamp_squared_error": 0,
            "screen_timestamp_gt_values": [],
            "screen_timestamp_absolute_errors": [],
            "screen_timestamp_signed_errors": [],
        }

    @staticmethod
    def _determine_event_type(event: McapMessage) -> str:
        """Determine the event type from a decoded event object."""
        event_type = event.topic
        # Handle mouse events with special categorization
        if event_type == "mouse/raw":
            if event.decoded.button_flags != 0:
                return "mouse_op"
            else:
                return "mouse_nop"
        # Handle other event types - can be extended here
        if event_type in ["keyboard", "screen"]:
            return event_type
        raise ValueError(f"Unknown event type: {event_type}")

    def _process_keyboard_event(self, pred_event: McapMessage, gt_event: McapMessage) -> None:
        """Process keyboard event metrics."""
        pred_decoded = pred_event.decoded
        gt_decoded = gt_event.decoded
        # Only count as correct if both vk and event_type are correct
        both_correct = (pred_decoded.vk == gt_decoded.vk) and (pred_decoded.event_type == gt_decoded.event_type)
        self.metrics["keyboard_accuracy_correct"] += int(both_correct)
        self.metrics["keyboard_accuracy_total"] += 1

    def _process_mouse_move_metrics(self, pred_event: McapMessage, gt_event: McapMessage) -> None:
        """Process mouse movement metrics (R², percentage errors, signed PE)."""
        pred_decoded = pred_event.decoded
        gt_decoded = gt_event.decoded

        # Compute squared Euclidean distance between predicted and ground truth vectors
        dx_error = pred_decoded.dx - gt_decoded.dx
        dy_error = pred_decoded.dy - gt_decoded.dy
        self.metrics["mouse_move_squared_error"] += dx_error**2 + dy_error**2
        self.metrics["mouse_move_gt_values"].append((gt_decoded.dx, gt_decoded.dy))

        # Compute percentage error using Euclidean distance between vectors
        euclidean_error = np.sqrt((pred_decoded.dx - gt_decoded.dx) ** 2 + (pred_decoded.dy - gt_decoded.dy) ** 2)
        gt_norm = np.sqrt(gt_decoded.dx**2 + gt_decoded.dy**2)
        if gt_norm > 0:  # Avoid division by zero
            percentage_error_euclidean = euclidean_error / gt_norm * 100
            self.metrics["mouse_move_pe_euclidean"].append(percentage_error_euclidean)

        # Compute signed percentage error for individual x and y coordinates (pred-gt)/gt
        if gt_decoded.dx != 0:  # Avoid division by zero
            signed_pe_x = (pred_decoded.dx - gt_decoded.dx) / gt_decoded.dx * 100
            self.metrics["mouse_move_signed_pe_x"].append(signed_pe_x)
        if gt_decoded.dy != 0:  # Avoid division by zero
            signed_pe_y = (pred_decoded.dy - gt_decoded.dy) / gt_decoded.dy * 100
            self.metrics["mouse_move_signed_pe_y"].append(signed_pe_y)

        # Compute movement direction error (angular error between movement vectors)
        direction_error = movement_direction_error_degrees(
            pred_decoded.dx, pred_decoded.dy, gt_decoded.dx, gt_decoded.dy
        )
        if direction_error is not None:
            self.metrics["mouse_move_direction_errors"].append(direction_error)

        # Compute mouse move accuracy at different precision levels
        # Check if predicted and ground truth coordinates match when rounded to different precision levels
        for precision in [1000, 100, 10, 1]:
            pred_x_rounded = pred_decoded.dx // precision
            pred_y_rounded = pred_decoded.dy // precision
            gt_x_rounded = gt_decoded.dx // precision
            gt_y_rounded = gt_decoded.dy // precision

            is_correct = (pred_x_rounded == gt_x_rounded) and (pred_y_rounded == gt_y_rounded)
            self.metrics[f"mouse_move_accuracy_{precision}_correct"] += int(is_correct)
            self.metrics[f"mouse_move_accuracy_{precision}_total"] += 1

    def _process_mouse_action_metrics(self, pred_event: McapMessage, gt_event: McapMessage) -> None:
        """Process mouse action metrics (button flags and scroll data)."""
        pred_decoded = pred_event.decoded
        gt_decoded = gt_event.decoded

        if gt_decoded.button_flags != 0:
            self.metrics["mouse_action_correct"] += int(pred_decoded.button_flags == gt_decoded.button_flags)
            self.metrics["mouse_action_total"] += 1
        if gt_decoded.button_data != 0:
            self.metrics["mouse_scroll_correct"] += int(pred_decoded.button_data == gt_decoded.button_data)
            self.metrics["mouse_scroll_total"] += 1

    def _process_timestamp_metrics(self, pred_event: McapMessage, gt_event: McapMessage, event_type: str) -> None:
        """Process timestamp metrics (R², absolute errors, and signed errors for quartiles)."""
        timestamp_error = pred_event.timestamp - gt_event.timestamp
        timestamp_abs_error = abs(timestamp_error)

        # Global timestamp metrics
        self.metrics["timestamp_squared_error"] += timestamp_error**2
        self.metrics["timestamp_gt_values"].append(gt_event.timestamp)
        self.metrics["timestamp_pred_values"].append(pred_event.timestamp)
        self.metrics["timestamp_absolute_errors"].append(timestamp_abs_error)
        self.metrics["timestamp_signed_errors"].append(timestamp_error)

        # Per-event-type timestamp metrics
        if event_type in ["keyboard", "mouse_op", "mouse_nop", "screen"]:
            self.metrics[f"{event_type}_timestamp_squared_error"] += timestamp_error**2
            self.metrics[f"{event_type}_timestamp_gt_values"].append(gt_event.timestamp)
            self.metrics[f"{event_type}_timestamp_absolute_errors"].append(timestamp_abs_error)
            self.metrics[f"{event_type}_timestamp_signed_errors"].append(timestamp_error)

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = False) -> dict:
        """Placeholder metrics function for evaluation.
        Args:
            eval_pred (EvalPrediction): Evaluation predictions
            compute_result (bool): Whether to compute metrics. Given to True for last evaluation batch.
        """
        if isinstance(eval_pred.predictions, tuple):
            eval_pred.predictions = eval_pred.predictions[0]
        if isinstance(eval_pred.label_ids, tuple):
            eval_pred.label_ids = eval_pred.label_ids[0]

        logits = eval_pred.predictions  # [batch_size, seq_len, vocab_size]
        # WHY THE HELL THESE ARE NOT NUMPY ARRAYS??? TYPE HINT LIEING
        assert isinstance(logits, torch.Tensor)  # can be bf16
        # NOTE: logits must be argmax on GPU since logits are large and take very long time to transfer to CPU
        predictions = logits.argmax(-1)  # [batch_size, seq_len]

        labels = eval_pred.label_ids  # [batch_size, seq_len]
        assert isinstance(labels, torch.Tensor)

        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        assert isinstance(predictions, np.ndarray) and isinstance(labels, np.ndarray), (
            f"Expected numpy arrays for predictions and labels, got {type(predictions)} and {type(labels)}"
        )

        # Shift predictions and labels
        shift_predictions = predictions[..., :-1]
        shift_labels = labels[..., 1:]

        for predictions, labels in zip(shift_predictions, shift_labels):
            # Since predictions/labels is shifted-by-1, if the first token is not <EVENT_START>, add it.
            event_start_token_id = self.tokenizer.convert_tokens_to_ids("<EVENT_START>")
            event_end_token_id = self.tokenizer.convert_tokens_to_ids("<EVENT_END>")
            if labels[0] != event_start_token_id:
                predictions = np.insert(predictions, 0, event_start_token_id)
                labels = np.insert(labels, 0, event_start_token_id)

            self._prepare_visualize_example(predictions, labels)

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

                gt_event = gt_events[0]
                # Determine event types using the _determine_event_type method. TODO: use mouse_op/nop separated metric!
                try:
                    gt_event_type = self._determine_event_type(gt_event)
                except ValueError:
                    warnings.warn(f"GT event type must be known but it's unknown: {gt_event}")
                    continue

                if gt_event_type in ["keyboard", "mouse_op", "mouse_nop", "screen"]:
                    self.metrics[f"{gt_event_type}_total"] += 1

                # If not comparable, skip
                if not (len(pred_events) == len(gt_events) == 1):
                    continue

                pred_event = pred_events[0]
                try:
                    pred_event_type = self._determine_event_type(pred_event)  # noqa: F841
                except ValueError:
                    # pred event can be unknown. In that case we do not count for metrics.
                    continue

                # Use topic here instead of event_type to allow comparing mouse_nop and mouse_op
                if pred_event.topic != gt_event.topic:
                    continue

                # Count comparable events (overall and per-event type)
                self.metrics["comparable_events"] += 1
                if gt_event_type in ["keyboard", "mouse_op", "mouse_nop", "screen"]:
                    self.metrics[f"{gt_event_type}_comparable"] += 1

                # Process event-specific metrics
                if gt_event_type == "keyboard":
                    self._process_keyboard_event(pred_event, gt_event)

                if gt_event_type in ["mouse_op", "mouse_nop"]:
                    self._process_mouse_move_metrics(pred_event, gt_event)
                    self._process_mouse_action_metrics(pred_event, gt_event)

                # Process timestamp metrics for all event types
                self._process_timestamp_metrics(pred_event, gt_event, gt_event_type)

        # Return metrics only for the last batch
        if compute_result:
            final_metrics = self._compute_final_metrics()
            self._initialize_metrics()  # Reset metrics for next evaluation
            self._visualize_examples()
            return final_metrics

        return {}

    def _compute_final_metrics(self) -> dict:
        """Compute final metrics from accumulated values."""
        final_metrics = {}

        # Calculate comparable events ratio
        if self.metrics["total_events"] > 0:
            final_metrics["comparable"] = self.metrics["comparable_events"] / self.metrics["total_events"]

        # Calculate per-event type comparable ratios and event type ratios
        total_events_by_type = sum(
            self.metrics[f"{event_type}_total"] for event_type in ["keyboard", "mouse_op", "mouse_nop", "screen"]
        )

        for event_type in ["keyboard", "mouse_op", "mouse_nop", "screen"]:
            total_key = f"{event_type}_total"
            comparable_key = f"{event_type}_comparable"

            # Comparable ratio for this event type
            if self.metrics[total_key] > 0:
                final_metrics[f"{event_type}_comparable"] = self.metrics[comparable_key] / self.metrics[total_key]

            # Event type ratio (proportion of total events)
            if total_events_by_type > 0:
                final_metrics[f"{event_type}_ratio"] = self.metrics[total_key] / total_events_by_type

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

            # Calculate mouse move accuracy metrics at different precision levels
            for precision in [1000, 100, 10, 1]:
                total_key = f"mouse_move_accuracy_{precision}_total"
                correct_key = f"mouse_move_accuracy_{precision}_correct"
                if self.metrics[total_key] > 0:
                    final_metrics[f"mouse_move_accuracy_{precision}"] = (
                        self.metrics[correct_key] / self.metrics[total_key]
                    )

            # Calculate R² and percentile metrics for mouse movement (2D vector)
            if len(self.metrics["mouse_move_gt_values"]) > 0:
                gt_vectors = np.array(self.metrics["mouse_move_gt_values"])  # Shape: (n_samples, 2)

                # R² = 1 - (SS_res / SS_tot)
                # SS_res = sum of squared residuals (already computed as Euclidean distances)
                # SS_tot = sum of squared deviations from mean vector
                mean_vector = np.mean(gt_vectors, axis=0)  # Shape: (2,)

                # Compute total sum of squares for the 2D vector
                ss_tot = np.sum(np.sum((gt_vectors - mean_vector) ** 2, axis=1))

                if ss_tot > 0:
                    final_metrics["mouse_move_r2"] = 1 - (self.metrics["mouse_move_squared_error"] / ss_tot)
                else:
                    final_metrics["mouse_move_r2"] = 1.0  # Perfect prediction if no variance

                # Calculate percentile-based percentage error metrics for Euclidean distance
                if len(self.metrics["mouse_move_pe_euclidean"]) > 0:
                    pe_values_euclidean = np.array(self.metrics["mouse_move_pe_euclidean"])
                    final_metrics["mouse_move_pe_euclidean_p0"] = np.percentile(pe_values_euclidean, 0)
                    final_metrics["mouse_move_pe_euclidean_p25"] = np.percentile(pe_values_euclidean, 25)
                    final_metrics["mouse_move_pe_euclidean_p50"] = np.percentile(pe_values_euclidean, 50)
                    final_metrics["mouse_move_pe_euclidean_p75"] = np.percentile(pe_values_euclidean, 75)
                    final_metrics["mouse_move_pe_euclidean_p95"] = np.percentile(pe_values_euclidean, 95)
                    final_metrics["mouse_move_pe_euclidean_p100"] = np.percentile(pe_values_euclidean, 100)

                # Calculate IQM with 95% stratified bootstrap CIs for X coordinate signed PE
                if len(self.metrics["mouse_move_signed_pe_x"]) > 0:
                    signed_pe_values_x = np.array(self.metrics["mouse_move_signed_pe_x"])
                    iqm_x = interquartile_mean(signed_pe_values_x)
                    ci_lower_x, ci_upper_x = stratified_bootstrap_ci(signed_pe_values_x)
                    final_metrics["mouse_move_signed_pe_x_iqm"] = iqm_x
                    final_metrics["mouse_move_signed_pe_x_ci_lower"] = ci_lower_x
                    final_metrics["mouse_move_signed_pe_x_ci_upper"] = ci_upper_x

                # Calculate IQM with 95% stratified bootstrap CIs for Y coordinate signed PE
                if len(self.metrics["mouse_move_signed_pe_y"]) > 0:
                    signed_pe_values_y = np.array(self.metrics["mouse_move_signed_pe_y"])
                    iqm_y = interquartile_mean(signed_pe_values_y)
                    ci_lower_y, ci_upper_y = stratified_bootstrap_ci(signed_pe_values_y)
                    final_metrics["mouse_move_signed_pe_y_iqm"] = iqm_y
                    final_metrics["mouse_move_signed_pe_y_ci_lower"] = ci_lower_y
                    final_metrics["mouse_move_signed_pe_y_ci_upper"] = ci_upper_y

                # Calculate movement direction error percentiles
                if len(self.metrics["mouse_move_direction_errors"]) > 0:
                    direction_errors = np.array(self.metrics["mouse_move_direction_errors"])
                    final_metrics["mouse_move_direction_error_p50"] = np.percentile(direction_errors, 50)
                    final_metrics["mouse_move_direction_error_p95"] = np.percentile(direction_errors, 95)

            # Calculate timestamp metrics (R², percentiles of absolute errors, and percentiles of signed errors)
            if len(self.metrics["timestamp_gt_values"]) > 0:
                gt_timestamps = np.array(self.metrics["timestamp_gt_values"])

                # Calculate R² for timestamps
                mean_gt_timestamp = np.mean(gt_timestamps)
                ss_tot_timestamp = np.sum((gt_timestamps - mean_gt_timestamp) ** 2)

                if ss_tot_timestamp > 0:
                    final_metrics["timestamp_r2"] = 1 - (self.metrics["timestamp_squared_error"] / ss_tot_timestamp)
                else:
                    final_metrics["timestamp_r2"] = 1.0  # Perfect prediction if no variance

            if len(self.metrics["timestamp_absolute_errors"]) > 0:
                abs_errors = np.array(self.metrics["timestamp_absolute_errors"])

                # Calculate percentiles for absolute errors
                final_metrics["timestamp_abs_error_p0"] = np.percentile(abs_errors, 0)
                final_metrics["timestamp_abs_error_p25"] = np.percentile(abs_errors, 25)
                final_metrics["timestamp_abs_error_p50"] = np.percentile(abs_errors, 50)
                final_metrics["timestamp_abs_error_p75"] = np.percentile(abs_errors, 75)
                final_metrics["timestamp_abs_error_p95"] = np.percentile(abs_errors, 95)
                final_metrics["timestamp_abs_error_p100"] = np.percentile(abs_errors, 100)

            if len(self.metrics["timestamp_signed_errors"]) > 0:
                signed_errors = np.array(self.metrics["timestamp_signed_errors"])

                # Calculate IQM with 95% stratified bootstrap CIs for signed errors
                iqm_signed = interquartile_mean(signed_errors)
                ci_lower_signed, ci_upper_signed = stratified_bootstrap_ci(signed_errors)
                final_metrics["timestamp_signed_error_iqm"] = iqm_signed
                final_metrics["timestamp_signed_error_ci_lower"] = ci_lower_signed
                final_metrics["timestamp_signed_error_ci_upper"] = ci_upper_signed

            # Calculate per-event-type timestamp metrics
            for event_type in ["keyboard", "mouse_op", "mouse_nop", "screen"]:
                gt_values_key = f"{event_type}_timestamp_gt_values"
                squared_error_key = f"{event_type}_timestamp_squared_error"
                abs_errors_key = f"{event_type}_timestamp_absolute_errors"
                signed_errors_key = f"{event_type}_timestamp_signed_errors"

                # Calculate R² for this event type
                if len(self.metrics[gt_values_key]) > 0:
                    gt_timestamps = np.array(self.metrics[gt_values_key])
                    mean_gt_timestamp = np.mean(gt_timestamps)
                    ss_tot_timestamp = np.sum((gt_timestamps - mean_gt_timestamp) ** 2)

                    if ss_tot_timestamp > 0:
                        final_metrics[f"{event_type}_timestamp_r2"] = 1 - (
                            self.metrics[squared_error_key] / ss_tot_timestamp
                        )
                    else:
                        final_metrics[f"{event_type}_timestamp_r2"] = 1.0  # Perfect prediction if no variance

                # Calculate percentiles for absolute errors
                if len(self.metrics[abs_errors_key]) > 0:
                    abs_errors = np.array(self.metrics[abs_errors_key])
                    final_metrics[f"{event_type}_timestamp_abs_error_p0"] = np.percentile(abs_errors, 0)
                    final_metrics[f"{event_type}_timestamp_abs_error_p25"] = np.percentile(abs_errors, 25)
                    final_metrics[f"{event_type}_timestamp_abs_error_p50"] = np.percentile(abs_errors, 50)
                    final_metrics[f"{event_type}_timestamp_abs_error_p75"] = np.percentile(abs_errors, 75)
                    final_metrics[f"{event_type}_timestamp_abs_error_p95"] = np.percentile(abs_errors, 95)
                    final_metrics[f"{event_type}_timestamp_abs_error_p100"] = np.percentile(abs_errors, 100)

                # Calculate IQM with 95% stratified bootstrap CIs for signed errors
                if len(self.metrics[signed_errors_key]) > 0:
                    signed_errors = np.array(self.metrics[signed_errors_key])
                    iqm_signed = interquartile_mean(signed_errors)
                    ci_lower_signed, ci_upper_signed = stratified_bootstrap_ci(signed_errors)
                    final_metrics[f"{event_type}_timestamp_signed_error_iqm"] = iqm_signed
                    final_metrics[f"{event_type}_timestamp_signed_error_ci_lower"] = ci_lower_signed
                    final_metrics[f"{event_type}_timestamp_signed_error_ci_upper"] = ci_upper_signed

        return final_metrics

    def _prepare_visualize_example(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        """Prepare predictions and labels for visualization."""
        self.visualize_queue.append((predictions, labels))

    def _visualize_examples(self) -> None:
        """Write decoded predictions and ground truth to files for inspection."""
        if len(self.visualize_queue) == 0:
            return

        for idx, (predictions, labels) in enumerate(self.visualize_queue[: self.eval_samples_to_show]):
            assert predictions.shape == labels.shape, (
                f"Predictions and labels have different shapes: {predictions.shape} != {labels.shape}"
            )
            mask = labels != -100
            predictions = predictions[mask]
            labels = labels[mask]

            accuracy = (predictions == labels).mean()

            # Decode predictions and labels with self.tokenizer
            pred_tokens = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)
            gt_tokens = self.tokenizer.batch_decode(labels, skip_special_tokens=False)

            # Write to files
            with open(os.path.join(self.output_dir, f"eval_step_{self.state.global_step}.md"), "a") as f:
                f.write(f"Sample {idx}\n")
                f.write(f"Accuracy: {accuracy:.2f}\n")
                f.write("Predictions:\n")
                f.write(f"{''.join(pred_tokens)}\n")
                f.write("Ground Truth:\n")
                f.write(f"{''.join(gt_tokens)}\n")
                f.write(f"{'-' * 80}\n")

            # pickle sample for debug
            with open(os.path.join(self.output_dir, f"eval_step_{self.state.global_step}_{idx:03d}.pkl"), "wb") as f:
                pickle.dump({"predictions": predictions, "labels": labels}, f)


def print_evaluation_results(
    metrics: dict[str, float], title: str = "Evaluation Results", file: Optional[TextIO] = None
) -> None:
    """
    Print evaluation metrics in a human-readable format.

    Args:
        metrics: Dictionary of evaluation metrics from OWAEvaluatorBatched
        title: Title to display at the top of the results
        file: Optional file object to write to (default: None, prints to stdout)
    """
    print(f"\n{'=' * 60}", file=file)
    print(f"{title:^60}", file=file)
    print(f"{'=' * 60}", file=file)

    # Overall metrics
    print("\n📊 OVERALL METRICS", file=file)
    print(f"{'─' * 40}", file=file)
    if "comparable" in metrics:
        print(f"  Overall Comparable Events: {metrics['comparable']:.1%}", file=file)

    # Event type distribution
    print("\n📈 EVENT TYPE DISTRIBUTION", file=file)
    print(f"{'─' * 40}", file=file)
    event_types = ["keyboard", "mouse_op", "mouse_nop", "screen"]
    for event_type in event_types:
        ratio_key = f"{event_type}_ratio"
        comparable_key = f"{event_type}_comparable"

        if ratio_key in metrics:
            ratio = metrics[ratio_key]
            comparable = metrics.get(comparable_key, 0)
            print(
                f"  {event_type.replace('_', ' ').title():<12}: {ratio:>6.1%} of data, {comparable:>6.1%} comparable",
                file=file,
            )

    # Accuracy metrics
    print("\n🎯 ACCURACY METRICS", file=file)
    print(f"{'─' * 40}", file=file)
    accuracy_metrics = {
        "keyboard_accuracy": "Keyboard Events",
        "mouse_action_accuracy": "Mouse Actions",
        "mouse_scroll_accuracy": "Mouse Scroll",
    }

    for key, label in accuracy_metrics.items():
        if key in metrics:
            print(f"  {label:<15}: {metrics[key]:>12.3%}", file=file)

    # Mouse move accuracy metrics at different precision levels
    mouse_move_accuracy_keys = [k for k in metrics.keys() if k.startswith("mouse_move_accuracy_")]
    if mouse_move_accuracy_keys:
        print("  Mouse Move Accuracy:", file=file)
        for precision in [1000, 100, 10, 1]:
            key = f"mouse_move_accuracy_{precision}"
            if key in metrics:
                print(f"    ±{precision} pixels: {metrics[key]:>12.3%}", file=file)

    # Mouse movement metrics
    print("\n🖱️  MOUSE MOVE METRICS", file=file)
    print(f"{'─' * 40}", file=file)
    if "mouse_move_r2" in metrics:
        print(f"  R² Score: {metrics['mouse_move_r2']:>12.3f}", file=file)

    # Euclidean percentage errors
    euclidean_keys = [k for k in metrics.keys() if k.startswith("mouse_move_pe_euclidean_p")]
    if euclidean_keys:
        print("  Euclidean PE Percentiles:", file=file)
        percentiles = ["p0", "p25", "p50", "p75", "p95", "p100"]
        values = [metrics.get(f"mouse_move_pe_euclidean_{p}", 0) for p in percentiles]
        print(
            f"    P0: {values[0]:>12.3f}%  P25: {values[1]:>12.3f}%  P50: {values[2]:>12.3f}%  P75: {values[3]:>12.3f}%  P95: {values[4]:>12.3f}%  P100: {values[5]:>12.3f}%",
            file=file,
        )

    # X-coordinate signed PE
    x_pe_iqm_key = "mouse_move_signed_pe_x_iqm"
    if x_pe_iqm_key in metrics:
        print("  X-Coordinate Signed PE - IQM with 95% stratified bootstrap CIs:", file=file)
        iqm = metrics.get("mouse_move_signed_pe_x_iqm", 0)
        ci_lower = metrics.get("mouse_move_signed_pe_x_ci_lower", 0)
        ci_upper = metrics.get("mouse_move_signed_pe_x_ci_upper", 0)
        print(
            f"    IQM: {iqm:>12.3f}%  [95% CI: {ci_lower:>12.3f}%, {ci_upper:>12.3f}%]",
            file=file,
        )

    # Y-coordinate signed PE
    y_pe_iqm_key = "mouse_move_signed_pe_y_iqm"
    if y_pe_iqm_key in metrics:
        print("  Y-Coordinate Signed PE - IQM with 95% stratified bootstrap CIs:", file=file)
        iqm = metrics.get("mouse_move_signed_pe_y_iqm", 0)
        ci_lower = metrics.get("mouse_move_signed_pe_y_ci_lower", 0)
        ci_upper = metrics.get("mouse_move_signed_pe_y_ci_upper", 0)
        print(
            f"    IQM: {iqm:>12.3f}%  [95% CI: {ci_lower:>12.3f}%, {ci_upper:>12.3f}%]",
            file=file,
        )

    # Movement direction errors
    direction_error_p50_key = "mouse_move_direction_error_p50"
    direction_error_p95_key = "mouse_move_direction_error_p95"
    if direction_error_p50_key in metrics and direction_error_p95_key in metrics:
        print("  Movement Direction Error (degrees):", file=file)
        p50 = metrics.get("mouse_move_direction_error_p50", 0)
        p95 = metrics.get("mouse_move_direction_error_p95", 0)
        print(
            f"    P50: {p50:>12.1f}°  P95: {p95:>12.1f}°",
            file=file,
        )

    # Timestamp metrics
    print("\n⏰ TIMESTAMP METRICS", file=file)
    print(f"{'─' * 40}", file=file)
    if "timestamp_r2" in metrics:
        print(f"  R² Score: {metrics['timestamp_r2']:>12.3f}", file=file)

    # Absolute error percentiles
    abs_error_keys = [k for k in metrics.keys() if k.startswith("timestamp_abs_error_p")]
    if abs_error_keys:
        print("  Absolute Error Percentiles (ms):", file=file)
        percentiles = ["p0", "p25", "p50", "p75", "p95", "p100"]
        values = [metrics.get(f"timestamp_abs_error_{p}", 0) / 1_000_000 for p in percentiles]  # Convert ns to ms
        print(
            f"    P0: {values[0]:>12.3f}  P25: {values[1]:>12.3f}  P50: {values[2]:>12.3f}  P75: {values[3]:>12.3f}  P95: {values[4]:>12.3f}  P100: {values[5]:>12.3f}",
            file=file,
        )

    # Signed error IQM with 95% stratified bootstrap CIs
    signed_error_iqm_key = "timestamp_signed_error_iqm"
    if signed_error_iqm_key in metrics:
        print("  Signed Error - IQM with 95% stratified bootstrap CIs (ms):", file=file)
        iqm = metrics.get("timestamp_signed_error_iqm", 0) / 1_000_000  # Convert ns to ms
        ci_lower = metrics.get("timestamp_signed_error_ci_lower", 0) / 1_000_000  # Convert ns to ms
        ci_upper = metrics.get("timestamp_signed_error_ci_upper", 0) / 1_000_000  # Convert ns to ms
        print(
            f"    IQM: {iqm:>12.3f}  [95% CI: {ci_lower:>12.3f}, {ci_upper:>12.3f}]",
            file=file,
        )

        # Add interpretation for signed errors using IQM
        if abs(iqm) < 0.001:  # Less than 1μs (0.001ms)
            bias_interpretation = "No significant timing bias"
        elif iqm > 0:
            bias_interpretation = f"Model predicts {iqm:.3f}ms too late"
        else:
            bias_interpretation = f"Model predicts {abs(iqm):.3f}ms too early"
        print(f"    Interpretation: {bias_interpretation}", file=file)

    # Per-event-type timestamp metrics
    event_types = ["keyboard", "mouse_op", "mouse_nop", "screen"]
    event_type_names = {"keyboard": "Keyboard", "mouse_op": "Mouse Op", "mouse_nop": "Mouse NoOp", "screen": "Screen"}

    for event_type in event_types:
        r2_key = f"{event_type}_timestamp_r2"
        abs_error_keys = [k for k in metrics.keys() if k.startswith(f"{event_type}_timestamp_abs_error_p")]
        signed_error_iqm_key = f"{event_type}_timestamp_signed_error_iqm"

        # Only show section if we have data for this event type
        if r2_key in metrics or abs_error_keys or signed_error_iqm_key in metrics:
            print(f"\n⏰ {event_type_names[event_type].upper()} TIMESTAMP METRICS", file=file)
            print(f"{'─' * 40}", file=file)

            # R² Score
            if r2_key in metrics:
                print(f"  R² Score: {metrics[r2_key]:>12.3f}", file=file)

            # Absolute error percentiles
            if abs_error_keys:
                print("  Absolute Error Percentiles (ms):", file=file)
                percentiles = ["p0", "p25", "p50", "p75", "p95", "p100"]
                values = [metrics.get(f"{event_type}_timestamp_abs_error_{p}", 0) / 1_000_000 for p in percentiles]
                print(
                    f"    P0: {values[0]:>12.3f}  P25: {values[1]:>12.3f}  P50: {values[2]:>12.3f}  P75: {values[3]:>12.3f}  P95: {values[4]:>12.3f}  P100: {values[5]:>12.3f}",
                    file=file,
                )

            # Signed error IQM with 95% stratified bootstrap CIs
            if signed_error_iqm_key in metrics:
                print("  Signed Error - IQM with 95% stratified bootstrap CIs (ms):", file=file)
                iqm = metrics.get(f"{event_type}_timestamp_signed_error_iqm", 0) / 1_000_000
                ci_lower = metrics.get(f"{event_type}_timestamp_signed_error_ci_lower", 0) / 1_000_000
                ci_upper = metrics.get(f"{event_type}_timestamp_signed_error_ci_upper", 0) / 1_000_000
                print(
                    f"    IQM: {iqm:>12.3f}  [95% CI: {ci_lower:>12.3f}, {ci_upper:>12.3f}]",
                    file=file,
                )

                # Add interpretation for signed errors using IQM
                if abs(iqm) < 0.001:  # Less than 1μs (0.001ms)
                    bias_interpretation = "No significant timing bias"
                elif iqm > 0:
                    bias_interpretation = f"Model predicts {iqm:.3f}ms too late"
                else:
                    bias_interpretation = f"Model predicts {abs(iqm):.3f}ms too early"
                print(f"    Interpretation: {bias_interpretation}", file=file)

    print(f"\n{'=' * 60}\n", file=file)


class OWASFTTrainer(SFTTrainer):
    """Custom SFT Trainer that saves predicted sequences and ground truth during evaluation."""

    args: OWASFTConfig

    def __init__(self, args: OWASFTConfig, **kwargs):
        self._eval_output_dir = os.path.join(args.output_dir or "./output", "eval")
        os.makedirs(self._eval_output_dir, exist_ok=True)

        tokenizer = cast(PreTrainedTokenizer, kwargs["processing_class"].tokenizer)
        self._compute_metrics = OWAEvaluatorBatched(
            tokenizer=tokenizer, eval_samples_to_show=args.eval_samples_to_show, output_dir=self._eval_output_dir
        )
        kwargs = kwargs | dict(args=args, compute_metrics=self._compute_metrics)
        super().__init__(**kwargs)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        # we do not need messes except logits/labels
        ignore_keys = ignore_keys or []
        ignore_keys += ["last_hidden_state", "past_key_values", "hidden_states", "attentions", "image_hidden_states"]

        # pass trainer state to metrics function
        self._compute_metrics.state = self.state

        # Run standard evaluation
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        # Save prediction examples on main process
        if self.is_world_process_zero() and self.args.eval_samples_to_show > 0:
            self._save_predictions_and_ground_truth(output, metric_key_prefix=metric_key_prefix)

        return output

    def _save_predictions_and_ground_truth(self, output: EvalLoopOutput, metric_key_prefix: str = "eval"):
        """Save predictions and ground truth examples for inspection."""
        assert output.metrics is not None

        # Save evaluation results
        step = self.state.global_step

        metrics = {}
        # detach metric_key_prefix prefix from metric names
        for key in list(output.metrics.keys()):
            if key.startswith(f"{metric_key_prefix}_"):
                metrics[key[len(metric_key_prefix) + 1 :]] = output.metrics[key]

        with open(os.path.join(self._eval_output_dir, f"{metric_key_prefix}_step_{step}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        with open(os.path.join(self._eval_output_dir, f"{metric_key_prefix}_step_{step}_metrics.md"), "w") as f:
            print_evaluation_results(metrics, f"Step {step}", file=f)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/mnt/harbor/projects/owa/checkpoints/tmp")
    with open("/mnt/harbor/projects/owa/checkpoints/tmp/eval/eval_step_1032_000.pkl", "rb") as f:
        outputs = pickle.load(f)

    print(outputs)

    predictions = torch.from_numpy(outputs["predictions"])
    logits = torch.zeros((predictions.shape[0], tokenizer.vocab_size))
    # Set logit value
    for idx, pred in enumerate(predictions):
        logits[idx, pred] = 1
    logits = logits.unsqueeze(0)

    labels = torch.from_numpy(outputs["labels"][np.newaxis, ...])
    eval_prediction = EvalPrediction(predictions=logits, label_ids=labels)

    evaluator = OWAEvaluatorBatched(tokenizer=tokenizer, output_dir=".")
    metrics = evaluator(eval_prediction, compute_result=True)
    print(metrics)
