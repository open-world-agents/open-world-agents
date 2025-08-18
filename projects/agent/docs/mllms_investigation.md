# MLLMs Investigation

This document compares multimodal models based on 5 key questions:

1. Overall architecture (param count, target image size, tokens per image)
2. Image preprocessing pipeline (pad/crop/resize stages)
3. Multi-crop feature control (how to disable)
4. Multi-image batch handling (interleaved batches)
5. Image token structure (prefix/suffix tokens, token placement)

## SmolVLM2

### 1. Overall Architecture
**Model Type**: Vision-Language Model with SmolLM2 backbone

**Model Variants:**
- **SmolVLM2-256M**: ~256M parameters
- **SmolVLM2-500M**: ~500M parameters
- **SmolVLM2-2.2B**: ~2.2B parameters

**Architecture Components:**
- **Vision Encoder**: SmolVLMVisionTransformer (based on SigLIP)
- **Language Model**: SmolLM2 text decoder
- **Connector**: Perceiver resampler
- **Scale Factor**: 4 (pixel shuffle downsampling)

**Image Processing Configuration:**
- **Target Processing Size**: 2048px (longest edge)
- **Max Patch Size**: 512px (longest edge) after splitting
- **Vision Encoder Input**: 512×512 with 16×16 patch size
- **Tokens per Image**: 64 (fixed, via perceiver resampler)
- **Multi-patch Images**: Variable number of 512px patches + 1 global image patch

**Vision Configuration:**
- **Hidden Size**: 768
- **Patch Size**: 16
- **Image Size**: 512 (vision encoder input)
- **Perceiver Resampler**: 64 latents, 6 layers, 16 heads

### 2. Image Preprocessing Pipeline
```python
# Processing stages (SmolVLMImageProcessor):
1. Convert RGB (do_convert_rgb=True)
2. Resize to target size (longest_edge: 2048)
3. Image splitting (do_image_splitting=True)
4. Rescale (factor: 1/255)
5. Normalize (mean/std: [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
6. Pad to uniform dimensions (do_pad=True)
```

**Image Splitting Details:**
- **Algorithm**: Grid-based optimal splitting into patches
- **Max patch size**: 512px (longest edge) for all variants
- **Split calculation**: `ceil(height/512)` × `ceil(width/512)` grid
- **Global image**: Added as final patch (resized to 512×512)
- **Padding**: Zero-image padding to create uniform batch dimensions
- **Resampling**: LANCZOS interpolation
- **Output**: List of 512×512 patches + 1 global image patch

### 3. Disabling Multi-Crop
```python
# Method 1: Via processor call (Recommended)
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
inputs = processor(
    images=images,
    text=text,
    do_image_splitting=False,  # Disables multi-crop
    return_tensors="pt"
)

# Method 2: Modify image processor default
processor.image_processor.do_image_splitting = False
```

### 4. Multi-Image Batch Handling
- **Batch Structure**: `(batch_size, max_num_images, C, H, W)`
- **Padding Strategy**: Zero-image padding to maximum count per batch
- **Attention Handling**: Uses `pixel_attention_mask` to ignore padded regions
- **Memory Impact**: Less efficient due to padding overhead
- **Example**: Batch with [3,2,5] images → padded to [5,5,5] with zero images
- **Processing**: Each image split into patches, then batched with padding

### 5. Image Token Structure
```python
# Token definitions:
fake_image_token = "<fake_token_around_image>"  # Wrapper token
image_token = "<image>"                         # Core image token
global_image_token = "<global-img>"            # Global image marker
row_col_tokens = "<row_1_col_1>", etc.         # Patch position markers

# Structure for split images:
<fake_token_around_image><row_1_col_1><image><image>...<image>
<fake_token_around_image><row_1_col_2><image><image>...<image>
...
<fake_token_around_image><global-img><image><image>...<image><fake_token_around_image>

# Access via processor:
processor.fake_image_token     # "<fake_token_around_image>"
processor.image_token          # "<image>"
processor.global_image_token   # "<global-img>"
```

---

## InternVL3

### 1. Overall Architecture
**Model Type**: Multimodal model with Qwen2/LLaMA backbone

**Model Variants:**
- InternVL3-1B, InternVL3-2B, InternVL3-8B, InternVL3-9B, InternVL3-14B, InternVL3-38B, InternVL3-78B

**Architecture Components:**
- **Vision Encoder**: InternVLVisionModel (ViT-based)
- **Language Model**:
  - **Qwen2**: 1B, 2B, 8B, 14B, 38B, 78B variants
  - **LLaMA**: 9B variant only
- **Connector**: InternVLMultiModalProjector with spatial downsampling

**Image Processing Configuration:**
- **Image Size**: 448×448 pixels
- **Patch Size**: 14×14 pixels
- **Initial Patches**: 448/14 × 448/14 = 32×32 = 1024 patches
- **Downsampling**: downsample_ratio=0.5 → 16×16 = 256 final tokens
- **Tokens per Image**: 256 (consistent across all variants)


### 2. Image Preprocessing Pipeline
```python
# Processing stages (GotOcr2ImageProcessor):
1. Convert RGB (do_convert_rgb=True)
2. Resize to 448×448 (InternVL3 configuration)
3. Multi-crop (crop_to_patches=False by default, but can be enabled)
4. Rescale (factor: 1/255)
5. Normalize (ImageNet mean/std: [0.485, 0.456, 0.406],
              [0.229, 0.224, 0.225])
6. Flattened processing (no image-level padding)
```

**Multi-Crop Details:**
- **Implementation**: Uses `crop_to_patches` parameter in GotOcr2ImageProcessor
- **InternVL3 Default**: crop_to_patches=False (disabled by default)
- **Min patches**: 1 (minimum number of patches)
- **Max patches**: 12 (maximum number of patches)
- **Resampling**: BICUBIC interpolation
- **Patch Strategy**: Optimal tiled canvas arrangement based on aspect ratio
- **Thumbnail**: Adds thumbnail image when multiple patches are created

### 3. Disabling Multi-Crop
```python
# Method 1: Via processor call (Recommended)
processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")
inputs = processor(
    images=images,
    text=text,
    crop_to_patches=False,  # Disables multi-crop
    return_tensors="pt"
)

# Method 2: Modify default configuration
processor._defaults["images_kwargs"]["crop_to_patches"] = False

# Method 3: Custom image processor
from transformers import GotOcr2ImageProcessor
image_processor = GotOcr2ImageProcessor.from_pretrained(
    "OpenGVLab/InternVL3-1B-hf",
    crop_to_patches=False
)
```

### 4. Multi-Image Batch Handling
- **Batch Structure**: Flattened `(total_patches, C, H, W)`
- **Padding Strategy**: No image-level padding (more memory efficient)
- **Processing**: Sequential image processing and concatenation
- **Memory Impact**: More efficient than SmolVLM2, no padding overhead
- **Patch Tracking**: Uses `num_patches` array to track patches per image
- **Token Management**: Uses image tokens and placeholders in text sequence

### 5. Image Token Structure
```python
# Token definitions:
start_image_token = "<img>"           # Start marker
end_image_token = "</img>"            # End marker
image_token = "<IMG_CONTEXT>"         # Repeated content token (256 times)

# Structure for single image:
<img><IMG_CONTEXT><IMG_CONTEXT>...<IMG_CONTEXT></img>
# (256 <IMG_CONTEXT> tokens between <img> and </img>)

# Access via processor:
processor.start_image_token     # "<img>"
processor.end_image_token       # "</img>"
processor.image_token          # "<IMG_CONTEXT>"

# Configuration:
processor.image_seq_length     # 256 (tokens per image)
```

---

## Summary Comparison

| Feature | SmolVLM2 | InternVL3 |
|---------|----------|-----------|
| **Tokens per Image** | 64 (fixed, perceiver resampler) | 256 (fixed) |
| **Image Processor** | SmolVLMImageProcessor | GotOcr2ImageProcessor |
| **Multi-crop Default** | True (do_image_splitting) | True (crop_to_patches) |
| **Batch Padding** | Zero-image padding | No padding (flattened) |
| **Memory Efficiency** | Lower (due to padding) | Higher (no padding) |
| **Vision Encoder** | SigLIP-based | ViT-based |
| **Normalization** | ImageNet stats | OPENAI_CLIP stats |
| **Resampling** | LANCZOS | BICUBIC |
