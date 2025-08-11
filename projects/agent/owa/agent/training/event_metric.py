import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer

from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.data.encoders import create_encoder
from owa.data.episode_tokenizer import EpisodeTokenizerConfig

# Disable logger by default for library usage. If needed logger.enable("owa.agent.training.event_metric")
logger.disable("owa.agent.training.event_metric")

# TODO: make these configurable, load and save
episode_tokenizer_config = EpisodeTokenizerConfig(encoder_type="hierarchical")
event_encoder = create_encoder("hierarchical")


def compute_single_event_metrics(
    *,
    event_pred_logits: npt.NDArray[np.float32],
    event_pred_tokens: npt.NDArray[np.int64],
    event_gt_tokens: npt.NDArray[np.int64],
    tokenizer: PreTrainedTokenizer,
) -> dict:
    """Compute metrics for a single event comparison."""
    metrics = {}

    # Cross entropy loss
    metrics["loss_sum"] = metrics["loss_mean"] = F.cross_entropy(
        torch.tensor(event_pred_logits), torch.tensor(event_gt_tokens), reduction="sum"
    ).item()

    # Token-level accuracy
    mask = event_gt_tokens != -100
    event_pred_tokens_masked = event_pred_tokens[mask]
    event_gt_tokens_masked = event_gt_tokens[mask]
    metrics["accuracy"] = (event_pred_tokens_masked == event_gt_tokens_masked).mean().item()

    # Decode tokens to text
    event_pred_text = tokenizer.decode(event_pred_tokens_masked, skip_special_tokens=False)
    event_gt_text = tokenizer.decode(event_gt_tokens_masked, skip_special_tokens=False)

    # Replace image tokens with fake image placeholder
    repeated_image_pattern = f"{episode_tokenizer_config.image_token_prefix}{episode_tokenizer_config.image_token_suffix}"  # fmt: skip
    event_pred_text = event_pred_text.replace(repeated_image_pattern, episode_tokenizer_config.fake_image_placeholder)
    event_gt_text = event_gt_text.replace(repeated_image_pattern, episode_tokenizer_config.fake_image_placeholder)

    # Decode text to structured events
    event_pred = event_encoder.decode_batch([event_pred_text], suppress_errors=True)[0]
    event_gt = event_encoder.decode_batch([event_gt_text], suppress_errors=True)[0]

    # e.g. <EVENT_START><3><5><0><MOUSE><1><1><19><19><9><9><8><6><0><0><0><EVENT_END>
    logger.debug(f"{event_pred_text=}, {event_gt_text=}")

    # Handle invalid schema cases
    if event_pred is None or event_gt is None:
        metrics["comparable"] = False
        metrics["comparison_status"] = "invalid_schema"
        return metrics

    # Determine event types with graceful handling
    pred_type = _determine_event_type(event_pred)
    gt_type = _determine_event_type(event_gt)

    # Add type information and type-specific metrics
    metrics["pred_type"] = pred_type
    metrics["gt_type"] = gt_type
    metrics[f"{pred_type}_loss_sum"] = metrics["loss_sum"]
    metrics[f"{pred_type}_loss_mean"] = metrics["loss_mean"]
    metrics[f"{pred_type}_accuracy"] = metrics["accuracy"]

    # Check type compatibility
    if pred_type != gt_type:
        metrics["comparable"] = False
        metrics["comparison_status"] = "type_mismatch"
        return metrics

    metrics["comparable"] = True
    metrics["comparison_status"] = "valid"

    # TODO: do not hard-code these
    metrics["timestamp_loss_sum"] = metrics["timestamp_loss_mean"] = F.cross_entropy(
        torch.tensor(event_pred_logits[1:4]), torch.tensor(event_gt_tokens[1:4]), reduction="sum"
    ).item()
    metrics["timestamp_accuracy"] = (event_pred_tokens_masked[1:4] == event_gt_tokens_masked[1:4]).mean().item()
    metrics[f"{pred_type}_timestamp_loss_sum"] = metrics["timestamp_loss_sum"]
    metrics[f"{pred_type}_timestamp_loss_mean"] = metrics["timestamp_loss_mean"]
    metrics[f"{pred_type}_timestamp_accuracy"] = metrics["timestamp_accuracy"]

    if gt_type in ("mouse_nop", "mouse_op"):
        # TODO: do not hard-code these
        metrics[f"{pred_type}_movement_p1_loss_sum"] = metrics[f"{pred_type}_movement_p1_loss_mean"] = F.cross_entropy(
            torch.tensor(event_pred_logits[5 : 5 + (1) * 2]),
            torch.tensor(event_gt_tokens[5 : 5 + (1) * 2]),
            reduction="sum",
        ).item()
        metrics[f"{pred_type}_movement_p2_loss_sum"] = metrics[f"{pred_type}_movement_p2_loss_mean"] = F.cross_entropy(
            torch.tensor(event_pred_logits[5 : 5 + (1 + 1) * 2]),
            torch.tensor(event_gt_tokens[5 : 5 + (1 + 1) * 2]),
            reduction="sum",
        ).item()
        metrics[f"{pred_type}_movement_p3_loss_sum"] = metrics[f"{pred_type}_movement_p3_loss_mean"] = F.cross_entropy(
            torch.tensor(event_pred_logits[5 : 5 + (1 + 1 + 1) * 2]),
            torch.tensor(event_gt_tokens[5 : 5 + (1 + 1 + 1) * 2]),
            reduction="sum",
        ).item()
        metrics[f"{pred_type}_movement_p4_loss_sum"] = metrics[f"{pred_type}_movement_p4_loss_mean"] = F.cross_entropy(
            torch.tensor(event_pred_logits[5 : 5 + (1 + 1 + 1 + 1) * 2]),
            torch.tensor(event_gt_tokens[5 : 5 + (1 + 1 + 1 + 1) * 2]),
            reduction="sum",
        ).item()
        metrics[f"{pred_type}_movement_p1_accuracy"] = (
            (event_pred_tokens_masked[5 : 5 + (1) * 2] == event_gt_tokens_masked[5 : 5 + (1) * 2]).mean().item()
        )
        metrics[f"{pred_type}_movement_p2_accuracy"] = (
            (event_pred_tokens_masked[5 : 5 + (1 + 1) * 2] == event_gt_tokens_masked[5 : 5 + (1 + 1) * 2])
            .mean()
            .item()
        )
        metrics[f"{pred_type}_movement_p3_accuracy"] = (
            (event_pred_tokens_masked[5 : 5 + (1 + 1 + 1) * 2] == event_gt_tokens_masked[5 : 5 + (1 + 1 + 1) * 2])
            .mean()
            .item()
        )
        metrics[f"{pred_type}_movement_p4_accuracy"] = (
            (
                event_pred_tokens_masked[5 : 5 + (1 + 1 + 1 + 1) * 2]
                == event_gt_tokens_masked[5 : 5 + (1 + 1 + 1 + 1) * 2]
            )
            .mean()
            .item()
        )
        metrics[f"{pred_type}_dx_pe"] = (
            (event_pred.decoded.dx - event_gt.decoded.dx) / event_gt.decoded.dx if event_gt.decoded.dx else 0
        )
        metrics[f"{pred_type}_dy_pe"] = (
            (event_pred.decoded.dy - event_gt.decoded.dy) / event_gt.decoded.dy if event_gt.decoded.dy else 0
        )
        metrics[f"{pred_type}_euclidean_pe"] = (
            (
                np.linalg.norm(
                    [event_pred.decoded.dx, event_pred.decoded.dy]
                    - np.array([event_gt.decoded.dx, event_gt.decoded.dy])
                )
                / np.linalg.norm([event_gt.decoded.dx, event_gt.decoded.dy])
            ).item()
            if np.linalg.norm([event_gt.decoded.dx, event_gt.decoded.dy])
            else 0
        )

    if gt_type == "mouse_op":
        metrics["mouse_op_button_flags_loss_sum"] = metrics["mouse_op_button_flags_loss_mean"] = F.cross_entropy(
            torch.tensor(event_pred_logits[13 : 13 + 3]), torch.tensor(event_gt_tokens[13 : 13 + 3]), reduction="sum"
        ).item()
        metrics["mouse_op_button_flags_accuracy"] = event_pred.decoded.button_flags == event_gt.decoded.button_flags
        if event_gt.decoded.button_data != 0:
            metrics["mouse_op_button_data_loss_sum"] = metrics["mouse_op_button_data_loss_mean"] = F.cross_entropy(
                torch.tensor(event_pred_logits[16 : 16 + 1]),
                torch.tensor(event_gt_tokens[16 : 16 + 1]),
                reduction="sum",
            ).item()
            metrics["mouse_op_button_data_accuracy"] = event_pred.decoded.button_data == event_gt.decoded.button_data

    if gt_type == "keyboard":
        metrics["keyboard_action_loss_sum"] = metrics["keyboard_action_loss_mean"] = F.cross_entropy(
            torch.tensor(event_pred_logits[5 : 5 + 2]), torch.tensor(event_gt_tokens[5 : 5 + 2]), reduction="sum"
        ).item()
        metrics["keyboard_action_accuracy"] = (event_pred.decoded.vk == event_gt.decoded.vk) and (
            event_pred.decoded.event_type == event_gt.decoded.event_type
        )

    return metrics


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


def aggregate_event_metrics(metric_list: list) -> dict:
    """Aggregate metrics across all events."""
    if not metric_list:
        return {}

    # Collect all unique metric keys from all events
    all_metric_keys = set()
    for metric in metric_list:
        all_metric_keys.update(metric.keys())

    # Define aggregation strategies for different metric types
    # Keys that should be summed
    sum_metrics = {"loss_sum"}

    # Keys that should be averaged (normalized by number of events)
    average_metrics = {"comparable", "loss_mean"}

    # Keys that end with specific suffixes should be summed
    sum_suffix_patterns = {"_loss_sum"}

    # Keys that end with specific suffixes should be averaged (only when present)
    average_suffix_patterns = {"_accuracy", "_loss_mean"}

    # Keys that end with specific suffixes should be IQM. TODO: verify whether IQM is good aggregator
    iqm_suffix_patterns = {"_pe"}

    # Keys that should be ignored in aggregation (metadata)
    ignore_metrics = {"pred_type", "gt_type", "comparison_status"}

    aggregated_metrics = {}

    for key in sorted(all_metric_keys):
        if key in ignore_metrics:
            continue

        # Collect values for this key from all metrics that have it
        values = [metric[key] for metric in metric_list if key in metric]

        if not values:  # Skip if no values found
            continue

        # Determine aggregation strategy
        if key in average_metrics:
            aggregated_metrics[key] = sum(values) / len(metric_list)
        elif key in sum_metrics:
            aggregated_metrics[key] = sum(values)
        elif any(key.endswith(suffix) for suffix in sum_suffix_patterns):
            aggregated_metrics[key] = sum(values)
        elif any(key.endswith(suffix) for suffix in average_suffix_patterns):
            # Only include accuracy metrics if they are actually present
            aggregated_metrics[key] = sum(values) / len(values)
        elif any(key.endswith(suffix) for suffix in iqm_suffix_patterns):
            interquartile_values = np.percentile(values, [25, 75])
            aggregated_metrics[key] = np.mean(interquartile_values).item()

    return aggregated_metrics


def compute_metrics_for_events(
    *,
    logits: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int64],
    tokenizer: PreTrainedTokenizer,
) -> dict:
    """Given shifted predictions and labels, compute metrics."""
    predictions = logits.argmax(axis=-1)
    # TODO: EMA-like metrics which considers tokens given sufficient context is easier to predict.
    sequence_length = predictions.shape[0]  # noqa: F841

    # If the first token is not <EVENT_START>, add it.
    event_start_token_id = tokenizer.convert_tokens_to_ids("<EVENT_START>")
    event_end_token_id = tokenizer.convert_tokens_to_ids("<EVENT_END>")
    if labels[0] != event_start_token_id:
        predictions = np.insert(predictions, 0, event_start_token_id)
        labels = np.insert(labels, 0, event_start_token_id)

    # Extract event boundaries
    where_start = np.where(labels == event_start_token_id)[0]
    where_end = np.where(labels == event_end_token_id)[0] + 1
    assert len(where_start) == len(where_end), (
        f"Number of <EVENT_START> and <EVENT_END> must be equal, found {len(where_start)} starts and {len(where_end)} ends."
    )

    metric_list = []

    for start, end in zip(where_start, where_end):
        event_pred_logits = logits[start:end]
        event_pred_tokens = predictions[start:end]
        event_gt_tokens = labels[start:end]

        # Compute metrics for this single event
        metrics = compute_single_event_metrics(
            event_pred_logits=event_pred_logits,
            event_pred_tokens=event_pred_tokens,
            event_gt_tokens=event_gt_tokens,
            tokenizer=tokenizer,
        )

        metric_list.append(metrics)

    metrics = aggregate_event_metrics(metric_list)
    return metrics


if __name__ == "__main__":
    import dill as pickle
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/raid12/scratch/claude/checkpoints/pretrain_smolvlm-256m_csgo_0811_legacy"
    )

    outputs = torch.load(
        "/mnt/raid12/scratch/claude/checkpoints/pretrain_smolvlm-256m_csgo_0811/eval/output.pt", pickle_module=pickle
    )
    logits, labels = outputs["logits"], outputs["labels"]
    logits = logits[:, :-1]
    labels = labels[1:]

    metric = compute_metrics_for_events(logits=logits, labels=labels, tokenizer=tokenizer)
    print(metric)
