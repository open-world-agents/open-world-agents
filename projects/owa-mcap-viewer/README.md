**WARNING**: This projects needs large-scale refactoring!!!

## TODOs

[] non-local(in terms of server) file reading: especially, huggingface dataset
[] local(in terms of client) file reading

## Usage

1. Setup `EXPORT_PATH` environment variable.
    ```
    export EXPORT_PATH=(path-to-your-folder-containing-mcap-and-mkvs)
2. `vuv install`
3. `uvicorn owa_viewer:app --host 0.0.0.0 --port 7860 --reload`