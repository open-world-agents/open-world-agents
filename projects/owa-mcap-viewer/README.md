## Usage

1. Setup `EXPORT_PATH` environment variable. You may setup `.env` instead.
    ```
    export EXPORT_PATH=(path-to-your-folder-containing-mcap-and-mkvs)
    ```
2. `vuv install`
3. `uvicorn owa_viewer:app --host 0.0.0.0 --port 7860 --reload`


## Known issues to fix

- [] Raw mouse support
- [] Sometimes keyboard/mouse is not shown
- [] remote file support
    - just open (1) fileserver (2) viewer server and deal with all file operation in client side(web browser) with streaming sense.
    - this architecture enables remote file visualize(e.g. mkv/mcap on huggingface dataset)