"""
====================================================================
    Definition of Dataset for E2E Agent Training
====================================================================

"Dataset" may be defined as table containing following columns:
- Source file: Path to the raw data file (e.g., MCAP, etc.)
- Timestamp: Specific time point in the raw data file.
    This timestamp is regarded as the "now" time in the agent system.
- Sampling weight: Weight for sampling the data at this timestamp.
    Common data may have a lower sampling weight, while rare data may have a higher sampling weight.

====================================================================
    Dataset Processing Pipeline Overview - Spec Driven
====================================================================

                    +-----------------------+
                    |    Raw Data Files     |
                    |     (MCAP, etc.)      |
                    +-----------+-----------+
                                |
                                v
                  +-------------------------------+
                  |    Perception Sampling         |   Uses PerceptionSamplingSpec     <------+
                  | (OWAMcapPerceptionReader)      |   - Guarantees lower bound info          |
                  +---------------+---------------+                                           |
                                  |     Raw Perceptions                                       |
                                  v                                                           |
                  +-------------------------------+                                           |
                  |   Processing Pipeline         |   Uses PerceptionSamplingSpec             |
                  | (perception_to_conversation,  |   - Builds/arranges best model input  <---+
                  |  lazy_load_images,            |
                  |  apply_processor)             |
                  +---------------+---------------+
                                  |
                                  v
                    +------------------------+
                    |    Training Dataset    |
                    |       (MyDataset)      |
                    +------------------------+

    Shared Spec
    ───────────
        PerceptionSamplingSpec (core/spec.py)
            └─ Defines: topics/modalities, time window, sampling interval (uniform/discrete), etc.
            └─ Always passed to both the reader (perception sampling) and the processor (model input builder).

----------------------------------------------------------------------
Data Processing Cycle:
----------------------------------------------------------------------
1. Raw Data Files → Perception Sampling → Processing Pipeline → Training Samples
2. Same processing components used in both training and inference

----------------------------------------------------------------------
Key Components:
----------------------------------------------------------------------
1. Perception Sampling
    - Extracts raw perceptions from recorded data files at specific timestamps, driven by PerceptionSamplingSpec
    - OWAMcapPerceptionReader: Reads perception data from MCAP files
        • Receives PerceptionSamplingSpec (guarantees lower bound of info: ensures all events/images required by the spec are fetched)
2. Processing Pipeline
    - Converts raw perceptions into model-ready training samples, also utilizing PerceptionSamplingSpec
    - Steps/pipeline:
        • perception_to_conversation: Formats perceptions into the exact structure required, selects/arranges window & rate as dictated by the spec
        • lazy_load_images: Loads and prepares image data on demand
        • apply_processor: Applies model-specific preprocessing to prepare final input
3. Dataset Implementations
    - build_dataset: Creates dataset by iterating through valid timestamps
    - MyDataset: PyTorch Dataset implementation for training

----------------------------------------------------------------------
Core Design Principle:
----------------------------------------------------------------------
The dataset processing pipeline reuses the same utility functions and passes the same PerceptionSamplingSpec
used in the agent system during inference, ensuring consistency between training and production.
    - Eliminates discrepancies and mismatches in data processing
    - Reduces maintenance by maximizing code sharing
    - Ensures that model training data exactly matches inference-time inputs (window, rate, structure)

----------------------------------------------------------------------
Common Implementation Patterns:
----------------------------------------------------------------------
1. Direct Pipeline Reuse:
    ```python
    # In Agent System (Inference Time)
    pending_thought = (Pipe(conversation) | lazy_load_images | apply_processor).execute()

    # In Dataset Processing (Training Time)
    pending_thought = (
        Pipe([], [], current_perception, now=timestamp)
        | (lambda *args: perception_to_conversation(*args, spec=spec))
        | lazy_load_images
    ).execute()
    ```
2. Deferred Processing:
    - Some heavy processing steps (like apply_processor) can be deferred to batch collation
    - Example: `collate_fn(processor, batch)` applies processing to batched samples
    - This allows for more efficient batch processing when appropriate

3. Interval-based Sampling:
    - Training data is often selected from specific valid intervals in raw data files
    - This allows focusing on relevant segments and skipping irrelevant portions
    - Intervals can be determined by event markers, quality thresholds, or other criteria

----------------------------------------------------------------------
Component Roles & Spec Usage:
----------------------------------------------------------------------
- PerceptionSamplingSpec:
    • Single contract for perception sampling and model input building
- OWAMcapPerceptionReader:
    • Ensures all raw data needed is present (lower bound guarantee)
- perception_to_conversation (and pipeline):
    • Selects the right set, in the right window/rate, precisely as required by the spec (finalizes best model input)

----------------------------------------------------------------------
SUMMARY
----------------------------------------------------------------------
- Unified, explicit PerceptionSamplingSpec guarantees full pipeline consistency and correctness, with no "lost" or "mismatched" data between recorded files and the model's expected inputs.
"""
