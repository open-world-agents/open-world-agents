# OWA Dataset Implementation Plan

## 🎯 **Objective**
Integrate the new minimal dataset design (EventDataset, BinnedDataset) into the existing pipeline while preserving all functionality including transforms, FSLDataset integration, and the complete data flow.

## 📊 **Current Architecture Analysis**

### **Data Flow**
```
Raw MCAP → EventDataset → BinnedDataset → FSLDataset → VLA Training
   (1)         (2)           (3)           (4)
```

### **Transform Integration**
- **Event Dataset Transform**: `create_event_dataset_transform()` - converts binary MCAP messages to `encoded_event` + `image`
- **Binned Dataset Transform**: `create_binned_dataset_transform()` - converts state/action sequences to VLA format
- **FSLDataset**: Tokenization-aware packing with sequence handling

### **Key Requirements**
1. **Preserve HuggingFace compatibility** - transforms use `dataset.set_transform()`
2. **Maintain FSLDataset integration** - must work with tokenized event datasets
3. **Keep all existing functionality** - rate limiting, topic filtering, binning, etc.
4. **Add configuration persistence** - track processing parameters through pipeline

## 🏗️ **Implementation Strategy**

### **Phase 1: Enhanced Dataset Configuration**

#### **1.1 Update EventDatasetConfig**
```python
@dataclass
class EventDatasetConfig(OWADatasetConfig):
    dataset_type: DatasetType = DatasetType.EVENT
    # Processing parameters for reproducibility
    rate_settings: Optional[Dict[str, float]] = None  # {"screen": 20, "mouse": 60}
    keep_topics: Optional[List[str]] = None  # ["screen", "keyboard"]
    num_workers: int = 4
    # Source tracking
    source_train_dir: Optional[str] = None
    source_test_dir: Optional[str] = None
    test_percent: Optional[float] = None
    # Transform settings
    encoder_type: str = "hierarchical"
    load_images: bool = True
```

#### **1.2 Update BinnedDatasetConfig**
```python
@dataclass
class BinnedDatasetConfig(OWADatasetConfig):
    dataset_type: DatasetType = DatasetType.BINNED
    # Binning parameters
    fps: float = 10.0
    filter_empty_actions: bool = False
    bin_interval_ns: Optional[int] = None  # Computed from fps
    # Source tracking
    source_event_dataset: Optional[str] = None
    # Transform settings
    instruction: str = "Complete the computer task"
    encoder_type: str = "hierarchical"
    load_images: bool = True
```

### **Phase 2: Script Integration**

#### **2.1 Script 01: Raw MCAP → EventDataset**
**Changes:**
- Import new dataset classes
- Create EventDatasetConfig from CLI args
- Replace `datasets.DatasetDict` creation with `EventDataset` creation
- Use `EventDataset.save_to_disk()` for automatic config persistence

**Implementation:**
```python
# Create config from CLI arguments
event_config = EventDatasetConfig(
    mcap_root_directory=str(train_dir),
    dataset_type=DatasetType.EVENT,
    rate_settings=rate_settings,
    keep_topics=topics_to_keep,
    num_workers=num_workers,
    source_train_dir=str(train_dir),
    source_test_dir=str(test_dir) if test_dir else None,
    test_percent=test_percent if not test_dir else None,
)

# Create EventDataset instead of regular HF Dataset
train_dataset = EventDataset(
    arrow_table=train_data.data,
    info=train_data.info,
    owa_config=event_config
)

# Save with automatic config persistence
dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
dataset_dict.save_to_disk(output_dir)
```

#### **2.2 Script 02: EventDataset → BinnedDataset**
**Changes:**
- Use `load_owa_dataset()` for auto-detection
- Create BinnedDatasetConfig inheriting from EventDataset config
- Replace regular HF Dataset with BinnedDataset
- Use `BinnedDataset.save_to_disk()` for config persistence

**Implementation:**
```python
# Auto-detect and load EventDataset
event_dataset_dict = load_owa_dataset(input_dir)
event_config = event_dataset_dict["train"].owa_config

# Create binned config inheriting from event config
binned_config = BinnedDatasetConfig(
    mcap_root_directory=event_config.mcap_root_directory,
    fps=fps,  # From CLI
    filter_empty_actions=filter_empty_actions,  # From CLI
    source_event_dataset=str(input_dir),
    # Inherit transform settings
    encoder_type=event_config.encoder_type,
    load_images=event_config.load_images,
)

# Create BinnedDataset
binned_dataset = BinnedDataset(
    arrow_table=binned_data.data,
    info=binned_data.info,
    owa_config=binned_config
)
```

### **Phase 3: Transform Integration**

#### **3.1 Create owa.data.datasets module**
```python
# New module: owa/data/datasets.py
# Contains all dataset classes and configurations

# Clean main interface: owa/data/__init__.py
from .datasets import EventDataset, BinnedDataset, load_owa_dataset

__all__ = [
    "load_dataset", "create_encoder",
    "create_event_dataset_transform", "create_binned_dataset_transform",
    # Essential dataset classes only
    "EventDataset", "BinnedDataset", "load_owa_dataset"
]
```

#### **3.2 Enhanced Transform Functions**
**No changes needed** - transforms work with any HuggingFace Dataset, including our new classes.

#### **3.3 Smart Transform Configuration**
```python
# In README examples, use config-aware transforms
event_dataset = load_owa_dataset("/path/to/event/dataset")
config = event_dataset["train"].owa_config

# Use config settings for transform
transform = create_event_dataset_transform(
    encoder_type=config.encoder_type,
    load_images=config.load_images
)
event_dataset.set_transform(transform)
```

### **Phase 4: FSLDataset Integration**

#### **4.1 Validation Method**
```python
class FSLDataset:
    def __init__(self, dataset, config):
        # Validate input is EventDataset (after tokenization)
        if hasattr(dataset, 'owa_config') and dataset.owa_config:
            logger.info(f"Using EventDataset with config: {dataset.owa_config.dataset_type}")
        
        # Continue with existing logic
        super().__init__(dataset, config)
```

**No other changes needed** - FSLDataset already works with any HuggingFace Dataset.

### **Phase 5: CLI Argument Strategy**

#### **5.1 Hybrid Approach**
- **Keep all existing CLI arguments** for explicit control
- **Auto-inherit from previous configs** when available
- **CLI overrides config values** when specified

#### **5.2 Smart Defaults**
```python
# Script 02 example
if input_dir and Path(input_dir, "owa_config.json").exists():
    # Load previous config
    event_dataset = load_owa_dataset(input_dir)
    prev_config = event_dataset["train"].owa_config
    
    # Use previous settings as defaults, CLI args override
    fps = args.fps or getattr(prev_config, 'fps', 10.0)
    encoder_type = args.encoder_type or getattr(prev_config, 'encoder_type', 'hierarchical')
```

## ✅ **Implementation Benefits**

### **1. Full Backward Compatibility**
- Existing scripts work unchanged
- Existing data can be loaded
- All CLI arguments preserved

### **2. Enhanced Functionality**
- Automatic config persistence
- Smart parameter inheritance
- Type-safe dataset operations
- Auto-detection loading

### **3. Improved Developer Experience**
- Clear dataset types throughout pipeline
- Reproducible processing parameters
- Better error messages and validation

### **4. Minimal Code Changes**
- ~50 lines changed in script 01
- ~30 lines changed in script 02
- ~10 lines changed in __init__.py
- No changes to transforms or FSLDataset

## 🧪 **Testing Strategy**

### **1. Unit Tests**
- Config serialization/deserialization
- Dataset creation and loading
- Transform compatibility

### **2. Integration Tests**
- End-to-end pipeline: MCAP → EventDataset → BinnedDataset
- Config inheritance through pipeline
- FSLDataset integration

### **3. Backward Compatibility Tests**
- Load existing datasets
- Run existing scripts
- Verify output compatibility

## 📝 **Implementation Order**

1. **Update dataset.py** with enhanced configs
2. **Update script 01** with EventDataset integration
3. **Update script 02** with BinnedDataset integration
4. **Update __init__.py** exports
5. **Update README** examples
6. **Test end-to-end pipeline**
7. **Clean up temporary files**

This plan ensures minimal disruption while adding powerful new capabilities to the OWA data pipeline.
