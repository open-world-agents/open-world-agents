# Event Metrics Evaluation System

## Overview
Compares predicted vs ground truth events using MSE/RMSE for continuous values and accuracy for discrete values.

## Metric Structure

### 1. Per-Event Comparison Metrics

Each event pair `(predicted_event[i], ground_truth_event[i])` is evaluated:

```python
{
    "event_comparisons": [
        {
            "position": int,                    # Index in sequence
            "comparable": bool,                 # Can these events be meaningfully compared? (event type matches and conforms to schema)
            "comparison_status": str,           # "valid", "type_mismatch", "invalid_format", "missing_fields"

            "predicted_type": str,              # e.g., "mouse/raw", "keyboard"
            "ground_truth_type": str,
            "timestamp_error_ms": float | None, # None if not comparable

            # Event-specific metrics (nested by type, only if comparable)
            "mouse_metrics": {                  # Only present for comparable mouse events
                "dx_error": float,
                "dy_error": float,
                "euclidean_error": float,
                "loss": float,
            } | None,

            "keyboard_metrics": {               # Only present for comparable keyboard events
                "vk_match": bool,
                "action_match": bool,
                "combined_match": bool,
                "loss": float,
            } | None,

            "screen_metrics": {                 # Only present for comparable screen events
                "loss": float,
            } | None,
        }
    ]
}
```

### 2. Sequence-Level Metrics (Aggregated from Per-Event)

```python
{
    # Basic counts
    "predicted_count": int,
    "ground_truth_count": int,
    "count_accuracy": float,                # 1.0 if counts match exactly
    "comparable_rate": float,               # Fraction of events that are comparable

    # Overall continuous metrics (aggregated across comparable events only)
    "timestamp_metrics": {
        "mse_ms": float,                    # MSE across comparable events
        "rmse_ms": float,                   # RMSE across comparable events
        "count": int,                       # Number of comparable events with timestamps
    },

    # Per-event-type continuous metrics (aggregated within comparable events of each type)
    "mouse_metrics": {
        "timestamp_mse_ms": float,          # MSE across comparable mouse events
        "timestamp_rmse_ms": float,         # RMSE across comparable mouse events
        "dx_mse": float,                    # MSE across comparable mouse events
        "dy_mse": float,                    # MSE across comparable mouse events
        "dx_rmse": float,                   # RMSE across comparable mouse events
        "dy_rmse": float,                   # RMSE across comparable mouse events
        "euclidean_mse": float,             # MSE of euclidean distance
        "euclidean_rmse": float,            # RMSE of euclidean distance
        "loss": float,                      # Total loss across comparable mouse events
        "comparable_count": int,            # Number of comparable mouse events
        "total_count": int,                 # Total number of mouse events (including non-comparable)
    },

    # Per-event-type discrete metrics (aggregated within comparable events of each type)
    "keyboard_metrics": {
        "timestamp_mse_ms": float,          # MSE across comparable keyboard events
        "timestamp_rmse_ms": float,         # RMSE across comparable keyboard events
        "vk_accuracy": float,               # Accuracy across comparable keyboard events
        "action_accuracy": float,           # Accuracy across comparable keyboard events
        "combined_accuracy": float,         # Both VK and action correct
        "loss": float,                      # Total loss across comparable keyboard events
        "comparable_count": int,            # Number of comparable keyboard events
        "total_count": int,                 # Total number of keyboard events (including non-comparable)
    },

    "screen_metrics": {
        "timestamp_mse_ms": float,          # MSE across comparable screen events
        "timestamp_rmse_ms": float,         # RMSE across comparable screen events
        "loss": float,                      # Total loss across comparable screen events
        "comparable_count": int,            # Number of comparable screen events
        "total_count": int,                 # Total number of screen events (including non-comparable)
    },
}
```

## Usage

```python
from owa.agent.training.event_metric import compute_metrics_for_event

# Compute metrics
metrics = compute_metrics_for_event(prediction_string, ground_truth_string)

# Access sequence-level metrics
print(f"Type accuracy: {metrics['type_accuracy']:.2f}")
print(f"Comparable rate: {metrics['comparable_rate']:.2f}")
print(f"Mouse RMSE: {metrics['mouse_metrics']['euclidean_rmse']:.2f}px")
print(f"Keyboard accuracy: {metrics['keyboard_metrics']['vk_accuracy']:.2f}")

# Access per-event details
for event in metrics['event_comparisons']:
    print(f"Event {event['position']}: {event['comparison_status']}")
```

## Key Features

- **MSE/RMSE** for continuous values (timestamps, mouse positions)
- **Accuracy** for discrete values (event types, keyboard keys)
- **Comparability tracking** before computing metrics
- **Error handling** for sequence length mismatch and decode failures
- **Event types supported**: mouse/raw, keyboard, screen

## Implementation

- **File**: `projects/agent/owa/agent/training/event_metric.py`
- **Main function**: `compute_metrics_for_event(prediction, ground_truth)`
- **Demo**: Run the file directly to see full working demonstration