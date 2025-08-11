# Event Metrics Design

## Overview

Compares predicted sequences and ground truth sequences. The implementation is done in two steps:

1. Compute event-by-event metrics
2. Aggregate metrics across all events

### Metric Types

1. Universal metric
- loss: cross-entropy loss

2. Regression metrics for continuous values
- PE(percent error): |predicted - ground_truth| / |ground_truth| * 100
- IQMPE(Interquartile Mean Percent Error): interquartile mean of PE
- precision_accuracy: sequential digit-level accuracy for hierarchical encodings
    - p1: first digit matches, p2: first AND second digits match, p3: first AND second AND third digits match
    - If earlier digit is wrong, all subsequent levels are 0
    - Number of levels depends on encoding (timestamp=3, mouse_delta=3, button_data=1)
    - Example: `<2><5><7>` vs `<2><3><7>` → p1=1, p2=0, p3=0

3. Binary classification metrics for discrete values
- precision, recall, accuracy: TP / (TP + FP), TP / (TP + FN), (TP + TN) / (TP + TN + FP + FN)
- f1-score: 2 * (precision * recall) / (precision + recall)

4. Multi-class classification metrics for categorical values
- accuracy: (TP + TN) / (TP + TN + FP + FN)

## Preliminaries

- tokenizer
- event encoder
- episode tokenizer

## Implementation

Implementation plan for `compute_metrics_for_event`:

1. For given labels and predictions, split all event-by-event. be aware of episode start and end tokens.
2. For each event, compute loss.
3. For each event, detokenize them with Tokenizer and then decode them with HierarchicalEventEncoder.
4. Compare the decoded GT and predicted events.
    - For timestamp compute PE and precision_accuracy. For unit of precision refer to config of event encoder. (e.g. HierarchicalEventEncoderConfig.timestamp_unit_ns and HierarchicalEventEncoderConfig.timestamp_bases)
    - For mouse compute PE and precision_accuracy for dx, dy, and euclidean distance. For unit of precision refer to config of event encoder. (e.g. HierarchicalEventEncoderConfig.mouse_delta_bases)
        - For button_flags, store GT/Pred per event, compute recall at sequence level
        - For button_data, store GT/Pred only when GT != 0, compute IQMPE and precision_accuracy at sequence level
        - Separate mouse events into two categories based on button_flags:
            - `mouse_nop`: events with zero button_flags (movement only)
            - `mouse_op`: events with non-zero button_flags (clicks, scrolls, etc.)
        - Compute separate metrics for mouse_nop and mouse_op categories
    - For keyboard compute accuracy for VK and action.
    - For screen compute nothing.
5. Aggregate the metrics across all events.
    - For PE, aggregate by IQM(InterQuartile Mean).
    - For precision_accuracy, aggregate by mean.
    - For accuracy, aggregate by mean.
    - For loss, aggregate by sum.

## Metric Structure

### 1. Per-Event Comparison Metrics

Each event pair `(predicted_event[i], ground_truth_event[i])` is evaluated:

```json
{
    "comparable": bool,                 // Can these events be meaningfully compared? (event type matches and conforms to schema)
    "comparison_status": str,           // "valid", "type_mismatch", "invalid_schema"
    "loss": float,
    "metrics": {
        "timestamp": {
            "gt": int,
            "pred": int,
            "metrics": ["IQMPE", "precision_accuracy"],
            "loss": float
        },
    }
}
```

### 2. Sequence-Level Metrics (Aggregated from Per-Event)

```json
{
    "loss": float,
    "metrics": {
        "timestamp": {
            "IQMPE": float,
            "precision_accuracy": float,
            "loss": float
        },
    }
}
```

