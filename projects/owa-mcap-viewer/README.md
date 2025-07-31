# OWA MCAP Viewer

A web-based viewer for Open World Agents MCAP files with flexible media reference support.

## Features

- **MediaRef Support**: Works with MCAP files containing embedded media, external video references, or traditional MKV pairs
- **Memory Efficient**: Handles large video files (10GB+) through streaming without loading into memory
- **Flexible Media Sources**: Supports data URIs, local files, remote URLs, and legacy MKV files
- **Interactive Visualization**: Real-time visualization of keyboard, mouse, and window events synchronized with video playback
- **Upload Support**: Upload MCAP files with or without separate video files
- **Validation**: Validate media references and check for missing or invalid media

## MediaRef Support

The viewer now supports the new MediaRef abstraction introduced in Open World Agents v0.5.0 with memory-efficient handling of large media files:

### Supported Media Types

1. **Embedded Media**: Images/videos encoded as data URIs within the MCAP file
2. **External Video Files**: References to video files (MKV, MP4, WebM) with frame timestamps
3. **External Images**: References to image files (PNG, JPG, GIF, WebP)
4. **Remote URLs**: HTTP/HTTPS URLs to media content
5. **Legacy MKV Pairs**: Traditional MCAP+MKV file pairs (backward compatibility)

### File Upload Options

- **MCAP Only (Recommended)**: Upload a single MCAP file with embedded or referenced media
- **MCAP + MKV Pair (Legacy)**: Upload separate MCAP and MKV files for older recordings

## Usage

1. Setup `EXPORT_PATH` environment variable. You may setup `.env` instead.
    ```bash
    export EXPORT_PATH=(path-to-your-folder-containing-mcap-files)
    ```
2. Install dependencies:
    ```bash
    vuv install
    ```
3. Run the server:
    ```bash
    uvicorn owa_viewer:app --host 0.0.0.0 --port 7860 --reload
    ```

## API Endpoints

### Media Serving
- `GET /files/media?mcap_filename=...&media_uri=...&local=true` - Serve media from MediaRef URI
- `GET /files/primary_video?mcap_filename=...&local=true` - Get primary video for playback
- `GET /files/validate_media?mcap_filename=...&local=true` - Validate media references

### File Management
- `GET /api/list_files?repo_id=...` - List MCAP files with media reference information
- `POST /upload/mcap` - Upload MCAP file with MediaRef support
- `POST /upload` - Upload legacy MCAP+MKV pair

## Testing

Run the MediaRef functionality test:
```bash
python test_mediaref_functionality.py
```

## Migration from Legacy Format

The viewer automatically detects and handles both new MediaRef format and legacy MCAP+MKV pairs. No migration is required - existing files will continue to work.