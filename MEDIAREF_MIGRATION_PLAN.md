# MediaRef Refactoring Migration Plan

## Overview
Complete migration plan for refactoring MediaRef from discriminated union (`EmbeddedRef | ExternalImageRef | ExternalVideoRef`) to unified URI-based design across the entire repository.

**Target Version**: v0.5.0 → v0.5.1
**Migration Script**: `v0_5_0_to_v0_5_1.py`

## Affected Files Inventory

### Core Implementation Files
- `projects/owa-msgs/owa/msgs/desktop/screen.py` - **PRIMARY**: MediaRef definitions and helper functions
- `projects/owa-msgs/tests/test_screen_msg.py` - **PRIMARY**: MediaRef tests
- `projects/owa-msgs/tests/test_screen_remote.py` - **PRIMARY**: Remote MediaRef tests

### Dependent Package Files
- `projects/owa-data/owa/data/transforms.py` - Uses ScreenCaptured.media_ref
- `projects/owa-data/owa/data/encoders/*.py` - Event encoders using ScreenCaptured
- `projects/owa-data/scripts/conversion/vpt_to_owamcap.py` - Data conversion scripts
- `projects/owa-env-gst/owa/env/gst/msg.py` - GST message handling
- `projects/owa-env-gst/owa/env/gst/omnimodal/appsink_recorder.py` - Recording functionality
- `projects/owa-env-gst/owa/env/gst/screen/*.py` - Screen capture components
- `projects/owa-core/tests/integration/test_message_system.py` - Integration tests
- `projects/ocap/owa/ocap/record.py` - Recording functionality
- `projects/mcap-owa-support/mcap_owa/hf_integration.py` - HuggingFace integration

### Migration Infrastructure
- `projects/owa-cli/owa/cli/mcap/migrate/migrators/v0_4_2_to_v0_5_0.py` - **REFERENCE**: Previous migration
- `projects/owa-cli/owa/cli/mcap/migrate/migrators/v0_5_0_to_v0_5_1.py` - **NEW**: Target migration script

## Breaking Changes Analysis

### 1. Type System Changes
```python
# OLD: Discriminated Union
MediaRef = Union[EmbeddedRef, ExternalImageRef, ExternalVideoRef]

# NEW: Single Class
class MediaRef(BaseModel):
    uri: str
    pts_ns: Optional[int] = None
```

### 2. Property Access Changes
```python
# OLD: Type-specific properties
if media_ref.type == "embedded":
    format = media_ref.format
    data = media_ref.data
elif media_ref.type == "external_image":
    path = media_ref.path

# NEW: Computed properties
format = media_ref.format  # Optional[str]
data = media_ref.data      # Optional[str] 
path = media_ref.path      # Optional[str]
```

### 3. JSON Serialization Changes
```python
# OLD JSON Format
{
  "type": "embedded",
  "format": "png", 
  "data": "iVBORw0KGgo..."
}

# NEW JSON Format  
{
  "uri": "data:image/png;base64,iVBORw0KGgo...",
  "pts_ns": null
}
```

### 4. Factory Method Changes
```python
# OLD: Type-specific constructors
EmbeddedRef(format="png", data="...")
ExternalImageRef(path="/path/to/image")
ExternalVideoRef(path="/path/to/video", pts_ns=123)

# NEW: Unified factory methods
MediaRef.from_embedded("png", "...")
MediaRef.from_path("/path/to/image")
MediaRef.from_path("/path/to/video", pts_ns=123)
```

## Migration Strategy

### Phase 1: Analysis Phase ✓
- [x] **1.1 Inventory Affected Files** - Complete file inventory
- [ ] **1.2 Analyze Import Dependencies** - Map import patterns
- [ ] **1.3 Identify Breaking Changes** - Document compatibility issues

### Phase 2: Core Implementation
- [ ] **2.1 Implement New MediaRef Class** - Create unified class with computed fields
- [ ] **2.2 Add Factory Methods** - Implement `from_embedded`, `from_path`, `from_file_uri`
- [ ] **2.3 Add Validation Logic** - URI validation for data URIs, URLs, file paths

### Phase 3: Update Loading Functions
- [ ] **3.1 Refactor _load_from_* Functions** - Update for new MediaRef
- [ ] **3.2 Update _get_media_info Function** - Refactor for new properties
- [ ] **3.3 Update _format_media_display Function** - Refactor display formatting
- [ ] **3.4 Update _compress_frame_to_embedded Function** - Return new MediaRef

### Phase 4: Update ScreenCaptured Class
- [ ] **4.1 Update ScreenCaptured.lazy_load** - Refactor loading logic
- [ ] **4.2 Update Type Checking Methods** - Update `has_embedded_data`, `has_external_reference`
- [ ] **4.3 Update Factory Methods** - Update `from_external_image`, `from_external_video`
- [ ] **4.4 Update embed_from_array Method** - Create new MediaRef
- [ ] **4.5 Update resolve_external_path Method** - Refactor path resolution

### Phase 5: Update Tests
- [ ] **5.1 Update test_screen_msg.py** - Refactor MediaRef tests
- [ ] **5.2 Update test_screen_remote.py** - Update remote tests
- [ ] **5.3 Add New MediaRef Tests** - Comprehensive new functionality tests

### Phase 6: Update Dependent Packages
- [ ] **6.1 Update owa-data Package** - transforms.py and encoders
- [ ] **6.2 Update owa-env-gst Package** - GST environment files
- [ ] **6.3 Update owa-core Package** - Integration tests
- [ ] **6.4 Update ocap Package** - Recording functionality
- [ ] **6.5 Update mcap-owa-support Package** - HuggingFace integration

### Phase 7: Create Migration Script
- [ ] **7.1 Create v0_5_0_to_v0_5_1.py** - Migration script implementation
- [ ] **7.2 Add Migration Logic** - Convert old to new MediaRef format
- [ ] **7.3 Test Migration Script** - Test with sample MCAP files

### Phase 8: Version Updates
- [ ] **8.1 Update owa-msgs to v0.5.1** - Bump version in pyproject.toml
- [ ] **8.2 Update Dependent Package Versions** - Update all dependent packages
- [ ] **8.3 Update Migration Script Dependencies** - Update migrator dependencies

### Phase 9: Integration Testing
- [ ] **9.1 Test New MediaRef Functionality** - Run all tests
- [ ] **9.2 Test Migration Script** - Test on real MCAP files
- [ ] **9.3 Test Cross-Package Integration** - Test package interactions

## Implementation Order

1. **Start with Core** (`owa-msgs/owa/msgs/desktop/screen.py`)
2. **Update Tests** (`owa-msgs/tests/`)
3. **Update Dependent Packages** (in dependency order)
4. **Create Migration Script** 
5. **Version Bumps and Integration Testing**

## Risk Mitigation

1. **Backward Compatibility**: Migration script handles all existing MCAP files
2. **Type Safety**: Computed fields provide same interface as old discriminated union
3. **Performance**: Computed fields are cached by Pydantic
4. **Testing**: Comprehensive test coverage for all scenarios

## Success Criteria

- [ ] All existing tests pass with new MediaRef
- [ ] Migration script successfully converts sample MCAP files
- [ ] All dependent packages work with new MediaRef
- [ ] Performance is maintained or improved
- [ ] API surface remains clean and intuitive
