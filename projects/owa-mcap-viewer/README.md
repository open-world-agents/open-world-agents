## Usage

1. Setup `EXPORT_PATH` environment variable. You may setup `.env` instead.
    ```
    export EXPORT_PATH=(path-to-your-folder-containing-mcap-and-mkvs)
    ```
2. `vuv install`
3. `uvicorn owa_viewer:app --host 0.0.0.0 --port 7860 --reload`