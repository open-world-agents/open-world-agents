import math
from typing import Any, Dict, List, Optional

import orjson

from mcap_owa.highlevel.mcap_msg import McapMessage
from owa.data.episode_tokenizer import EpisodeTokenizer


def _get_event_type(message: Optional[McapMessage]) -> Optional[str]:
    """Get the event type from a decoded message."""
    if message is None:
        return None
    return message.topic


def _compare_events(
    pred_msg: Optional[McapMessage],
    gt_msg: Optional[McapMessage],
    position: int,
) -> Dict[str, Any]:
    """Compare two events and return detailed comparison metrics."""
    comparison = {
        "position": position,
        "comparable": False,
        "comparison_status": "invalid_format",
        "predicted_type": _get_event_type(pred_msg),
        "ground_truth_type": _get_event_type(gt_msg),
        "timestamp_error_ms": None,
        "mouse_metrics": None,
        "keyboard_metrics": None,
        "screen_metrics": None,
    }

    # Check if both events are valid
    if pred_msg is None or gt_msg is None:
        if pred_msg is None and gt_msg is None:
            comparison["comparison_status"] = "both_invalid"
        elif pred_msg is None:
            comparison["comparison_status"] = "predicted_invalid"
        else:
            comparison["comparison_status"] = "ground_truth_invalid"
        return comparison

    # Check if event types match
    if pred_msg.topic != gt_msg.topic:
        comparison["comparison_status"] = "type_mismatch"
        return comparison

    # Events are comparable
    comparison["comparable"] = True
    comparison["comparison_status"] = "valid"

    # Calculate timestamp error
    timestamp_error_ns = abs(pred_msg.timestamp - gt_msg.timestamp)
    comparison["timestamp_error_ms"] = timestamp_error_ns / 1_000_000

    # Calculate event-specific metrics
    if pred_msg.topic in ["mouse", "mouse/raw"]:
        comparison["mouse_metrics"] = _compare_mouse_events(pred_msg, gt_msg)
    elif pred_msg.topic == "keyboard":
        comparison["keyboard_metrics"] = _compare_keyboard_events(pred_msg, gt_msg)
    elif pred_msg.topic == "screen":
        comparison["screen_metrics"] = _compare_screen_events(pred_msg, gt_msg)

    return comparison


def _compare_mouse_events(pred_msg: McapMessage, gt_msg: McapMessage) -> Dict[str, Any]:
    """Compare mouse events and return metrics."""

    pred_data = orjson.loads(pred_msg.message)
    gt_data = orjson.loads(gt_msg.message)

    # Extract mouse deltas
    pred_dx = pred_data.get("last_x", 0)
    pred_dy = pred_data.get("last_y", 0)
    gt_dx = gt_data.get("last_x", 0)
    gt_dy = gt_data.get("last_y", 0)

    # Calculate errors
    dx_error = abs(pred_dx - gt_dx)
    dy_error = abs(pred_dy - gt_dy)
    euclidean_error = math.sqrt(dx_error**2 + dy_error**2)

    return {
        "dx_error": float(dx_error),
        "dy_error": float(dy_error),
        "euclidean_error": float(euclidean_error),
    }


def _compare_keyboard_events(pred_msg: McapMessage, gt_msg: McapMessage) -> Dict[str, Any]:
    """Compare keyboard events and return metrics."""
    pred_data = orjson.loads(pred_msg.message)
    gt_data = orjson.loads(gt_msg.message)

    # Extract keyboard data
    pred_vk = pred_data.get("vk", 0)
    pred_action = pred_data.get("event_type", "")
    gt_vk = gt_data.get("vk", 0)
    gt_action = gt_data.get("event_type", "")

    # Calculate matches
    vk_match = pred_vk == gt_vk
    action_match = pred_action == gt_action
    combined_match = vk_match and action_match

    # Calculate loss (0 if match, 1 if no match)
    loss = 0.0 if combined_match else 1.0

    return {
        "vk_match": vk_match,
        "action_match": action_match,
        "combined_match": combined_match,
        "loss": loss,
    }


def _compare_screen_events(pred_msg: McapMessage, gt_msg: McapMessage) -> Dict[str, Any]:
    """Compare screen events and return metrics."""
    # For screen events, we mainly compare timestamps
    # The loss is 0 if timestamps match exactly, 1 otherwise
    timestamp_match = pred_msg.timestamp == gt_msg.timestamp
    loss = 0.0 if timestamp_match else 1.0

    return {
        "loss": loss,
    }


def _init_mouse_metrics() -> Dict[str, Any]:
    """Initialize empty mouse metrics structure."""
    return {
        "timestamp_mse_ms": 0.0,
        "timestamp_rmse_ms": 0.0,
        "dx_mse": 0.0,
        "dy_mse": 0.0,
        "dx_rmse": 0.0,
        "dy_rmse": 0.0,
        "euclidean_mse": 0.0,
        "euclidean_rmse": 0.0,
        "loss": 0.0,
        "comparable_count": 0,
        "total_count": 0,
    }


def _init_keyboard_metrics() -> Dict[str, Any]:
    """Initialize empty keyboard metrics structure."""
    return {
        "timestamp_mse_ms": 0.0,
        "timestamp_rmse_ms": 0.0,
        "vk_accuracy": 0.0,
        "action_accuracy": 0.0,
        "combined_accuracy": 0.0,
        "loss": 0.0,
        "comparable_count": 0,
        "total_count": 0,
    }


def _init_screen_metrics() -> Dict[str, Any]:
    """Initialize empty screen metrics structure."""
    return {
        "timestamp_mse_ms": 0.0,
        "timestamp_rmse_ms": 0.0,
        "loss": 0.0,
        "comparable_count": 0,
        "total_count": 0,
    }


def _calculate_sequence_metrics(
    event_comparisons: List[Dict[str, Any]],
    pred_messages: List[Optional[McapMessage]],
    gt_messages: List[Optional[McapMessage]],
) -> Dict[str, Any]:
    """Calculate sequence-level aggregated metrics."""
    # Basic counts (count both real messages and pseudo-messages)
    predicted_count = len([msg for msg in pred_messages if msg is not None])
    ground_truth_count = len([msg for msg in gt_messages if msg is not None])
    count_accuracy = 1.0 if predicted_count == ground_truth_count else 0.0

    # Count comparable events
    comparable_events = [comp for comp in event_comparisons if comp["comparable"]]
    comparable_rate = len(comparable_events) / len(event_comparisons) if event_comparisons else 0.0

    # Initialize metrics structure
    metrics = {
        "predicted_count": predicted_count,
        "ground_truth_count": ground_truth_count,
        "count_accuracy": count_accuracy,
        "comparable_rate": comparable_rate,
        "timestamp_metrics": {"mse_ms": 0.0, "rmse_ms": 0.0, "count": 0},
        "mouse_metrics": _init_mouse_metrics(),
        "keyboard_metrics": _init_keyboard_metrics(),
        "screen_metrics": _init_screen_metrics(),
    }

    if not comparable_events:
        return metrics

    # Calculate timestamp metrics across all comparable events
    timestamp_errors = [
        comp["timestamp_error_ms"] for comp in comparable_events if comp["timestamp_error_ms"] is not None
    ]
    if timestamp_errors:
        timestamp_mse = sum(error**2 for error in timestamp_errors) / len(timestamp_errors)
        metrics["timestamp_metrics"] = {
            "mse_ms": timestamp_mse,
            "rmse_ms": math.sqrt(timestamp_mse),
            "count": len(timestamp_errors),
        }

    # Calculate per-event-type metrics
    _calculate_mouse_sequence_metrics(comparable_events, metrics["mouse_metrics"])
    _calculate_keyboard_sequence_metrics(comparable_events, metrics["keyboard_metrics"])
    _calculate_screen_sequence_metrics(comparable_events, metrics["screen_metrics"])

    return metrics


def _calculate_mouse_sequence_metrics(comparable_events: List[Dict[str, Any]], mouse_metrics: Dict[str, Any]) -> None:
    """Calculate aggregated mouse metrics from comparable events."""
    mouse_events = [
        comp
        for comp in comparable_events
        if comp["predicted_type"] in ["mouse", "mouse/raw"] and comp["mouse_metrics"] is not None
    ]

    if not mouse_events:
        return

    # Extract individual metrics
    timestamp_errors = [comp["timestamp_error_ms"] for comp in mouse_events if comp["timestamp_error_ms"] is not None]
    dx_errors = [comp["mouse_metrics"]["dx_error"] for comp in mouse_events]
    dy_errors = [comp["mouse_metrics"]["dy_error"] for comp in mouse_events]
    euclidean_errors = [comp["mouse_metrics"]["euclidean_error"] for comp in mouse_events]

    # Calculate MSE and RMSE
    if timestamp_errors:
        timestamp_mse = sum(error**2 for error in timestamp_errors) / len(timestamp_errors)
        mouse_metrics["timestamp_mse_ms"] = timestamp_mse
        mouse_metrics["timestamp_rmse_ms"] = math.sqrt(timestamp_mse)

    if dx_errors:
        dx_mse = sum(error**2 for error in dx_errors) / len(dx_errors)
        mouse_metrics["dx_mse"] = dx_mse
        mouse_metrics["dx_rmse"] = math.sqrt(dx_mse)

    if dy_errors:
        dy_mse = sum(error**2 for error in dy_errors) / len(dy_errors)
        mouse_metrics["dy_mse"] = dy_mse
        mouse_metrics["dy_rmse"] = math.sqrt(dy_mse)

    if euclidean_errors:
        euclidean_mse = sum(error**2 for error in euclidean_errors) / len(euclidean_errors)
        mouse_metrics["euclidean_mse"] = euclidean_mse
        mouse_metrics["euclidean_rmse"] = math.sqrt(euclidean_mse)

    mouse_metrics["comparable_count"] = len(mouse_events)
    # Note: total_count would need to be calculated from all events, not just comparable ones


def _calculate_keyboard_sequence_metrics(
    comparable_events: List[Dict[str, Any]], keyboard_metrics: Dict[str, Any]
) -> None:
    """Calculate aggregated keyboard metrics from comparable events."""
    keyboard_events = [
        comp
        for comp in comparable_events
        if comp["predicted_type"] == "keyboard" and comp["keyboard_metrics"] is not None
    ]

    if not keyboard_events:
        return

    # Extract individual metrics
    timestamp_errors = [
        comp["timestamp_error_ms"] for comp in keyboard_events if comp["timestamp_error_ms"] is not None
    ]
    vk_matches = [comp["keyboard_metrics"]["vk_match"] for comp in keyboard_events]
    action_matches = [comp["keyboard_metrics"]["action_match"] for comp in keyboard_events]
    combined_matches = [comp["keyboard_metrics"]["combined_match"] for comp in keyboard_events]

    # Calculate metrics
    if timestamp_errors:
        timestamp_mse = sum(error**2 for error in timestamp_errors) / len(timestamp_errors)
        keyboard_metrics["timestamp_mse_ms"] = timestamp_mse
        keyboard_metrics["timestamp_rmse_ms"] = math.sqrt(timestamp_mse)

    if vk_matches:
        keyboard_metrics["vk_accuracy"] = sum(vk_matches) / len(vk_matches)

    if action_matches:
        keyboard_metrics["action_accuracy"] = sum(action_matches) / len(action_matches)

    if combined_matches:
        keyboard_metrics["combined_accuracy"] = sum(combined_matches) / len(combined_matches)

    keyboard_metrics["comparable_count"] = len(keyboard_events)


def _calculate_screen_sequence_metrics(
    comparable_events: List[Dict[str, Any]], screen_metrics: Dict[str, Any]
) -> None:
    """Calculate aggregated screen metrics from comparable events."""
    screen_events = [
        comp for comp in comparable_events if comp["predicted_type"] == "screen" and comp["screen_metrics"] is not None
    ]

    if not screen_events:
        return

    # Extract individual metrics
    timestamp_errors = [comp["timestamp_error_ms"] for comp in screen_events if comp["timestamp_error_ms"] is not None]

    # Calculate metrics
    if timestamp_errors:
        timestamp_mse = sum(error**2 for error in timestamp_errors) / len(timestamp_errors)
        screen_metrics["timestamp_mse_ms"] = timestamp_mse
        screen_metrics["timestamp_rmse_ms"] = math.sqrt(timestamp_mse)

    screen_metrics["comparable_count"] = len(screen_events)


def compute_metrics_for_event(prediction: str, ground_truth: str) -> dict:
    """Compute metrics for a single prediction."""

    episode_tokenizer = EpisodeTokenizer()
    pred_messages = episode_tokenizer.decode(prediction, suppress_errors=True)
    gt_messages = episode_tokenizer.decode(ground_truth, suppress_errors=True)

    # Skip first events since model can't expect it's timestamp
    pred_messages = pred_messages[1:]
    gt_messages = gt_messages[1:]

    print(pred_messages)
    print(gt_messages)

    # Pad shorter sequence with None values
    max_len = max(len(pred_messages), len(gt_messages))
    pred_messages.extend([None] * (max_len - len(pred_messages)))
    gt_messages.extend([None] * (max_len - len(gt_messages)))

    # Compare events pairwise
    event_comparisons = []
    for i, (pred_msg, gt_msg) in enumerate(zip(pred_messages, gt_messages)):
        comparison = _compare_events(pred_msg, gt_msg, i)
        event_comparisons.append(comparison)

    # Calculate sequence-level metrics
    sequence_metrics = _calculate_sequence_metrics(event_comparisons, pred_messages, gt_messages)

    return {"event_comparisons": event_comparisons, **sequence_metrics}


if __name__ == "__main__":
    PRED = "<EVENT_START><TIMESTAMP><7><8><0><MOUSE><9><10><0><0><0><0><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><2><1><2><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><1><2><MOUSE><10><10><0><0><0><0><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><2><1><5><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><3><7><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><5><0><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><6><2><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><7><5><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><8><7><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><8><3><MOUSE><10><10><0><0><2><0><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><0><0><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><3><0><0><MOUSE><10><10><0><0><0><0><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><0><6><MOUSE><10><10><0><0><0><0><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><1><2><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><3><1><2><MOUSE><10><10><2><0><0><0><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><1><8><MOUSE><10><10><0><9><0><8><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><2><5><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><3><2><5><MOUSE><10><9><3><9><0><6><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><3><1><MOUSE><65><press><EVENT_END><EVENT_START><TIMESTAMP><3><3><1><MOUSE><10><9><0><0><0><0><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><3><7><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><3><3><7><MOUSE><10><10><3><9><0><8><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><4><3><MOUSE><10><9><0><9><0><6><0><0><0><EVENT_END>"
    TRUE = "<EVENT_START><TIMESTAMP><2><0><6><MOUSE><10><10><1><0><0><2><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><2><1><2><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><1><2><MOUSE><10><10><0><0><2><0><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><2><2><5><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><3><7><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><5><0><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><6><2><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><7><5><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><8><7><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><2><9><3><MOUSE><10><10><0><0><4><0><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><0><0><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><3><0><0><MOUSE><10><10><1><0><0><0><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><0><6><MOUSE><10><10><2><0><0><0><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><1><2><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><3><1><2><MOUSE><10><10><1><0><0><0><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><1><8><MOUSE><10><9><1><9><0><8><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><2><5><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><3><2><5><MOUSE><10><9><1><9><0><8><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><3><1><KEYBOARD><65><release><EVENT_END><EVENT_START><TIMESTAMP><3><3><1><MOUSE><10><10><1><0><0><0><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><3><7><fake_token_around_image><global-img><fake_token_around_image><EVENT_END><EVENT_START><TIMESTAMP><3><3><7><MOUSE><10><9><1><9><0><8><0><0><0><EVENT_END><EVENT_START><TIMESTAMP><3><4><3><MOUSE><10><9><1><9><0><8><0><0><0><EVENT_END>"

    print("Computing event metrics...")
    metrics = compute_metrics_for_event(PRED, TRUE)

    print("\n=== Event Metrics Results ===")
    print(f"Predicted events: {metrics['predicted_count']}")
    print(f"Ground truth events: {metrics['ground_truth_count']}")
    print(f"Count accuracy: {metrics['count_accuracy']:.2f}")
    print(f"Comparable rate: {metrics['comparable_rate']:.2f}")

    print("\n=== Timestamp Metrics ===")
    ts = metrics["timestamp_metrics"]
    print(f"MSE: {ts['mse_ms']:.2f}ms, RMSE: {ts['rmse_ms']:.2f}ms (n={ts['count']})")

    print("\n=== Mouse Metrics ===")
    mouse = metrics["mouse_metrics"]
    print(f"Comparable events: {mouse['comparable_count']}")
    print(f"dx RMSE: {mouse['dx_rmse']:.2f}, dy RMSE: {mouse['dy_rmse']:.2f}")
    print(f"Euclidean RMSE: {mouse['euclidean_rmse']:.2f}")

    print("\n=== Keyboard Metrics ===")
    kb = metrics["keyboard_metrics"]
    print(f"Comparable events: {kb['comparable_count']}")
    if kb["comparable_count"] > 0:
        print(f"VK accuracy: {kb['vk_accuracy']:.2f}")
        print(f"Action accuracy: {kb['action_accuracy']:.2f}")
        print(f"Combined accuracy: {kb['combined_accuracy']:.2f}")

    print("\n=== Screen Metrics ===")
    screen = metrics["screen_metrics"]
    print(f"Comparable events: {screen['comparable_count']}")

    print("\n=== Sample Event Comparisons ===")
    for i, comp in enumerate(metrics["event_comparisons"][:5]):
        status = comp["comparison_status"]
        pred_type = comp["predicted_type"] or "None"
        gt_type = comp["ground_truth_type"] or "None"
        print(f"Event {i}: {status} - {pred_type} vs {gt_type}")
        if comp["timestamp_error_ms"] is not None:
            print(f"  Timestamp error: {comp['timestamp_error_ms']:.2f}ms")

    print("\nEvent metrics computation completed successfully!")
