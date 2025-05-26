# Viewer for OWAMcap

<div align="center">
  <img src="../viewer.png" alt="OWA Dataset Visualizer"/>
</div>

We provide a web-based viewer for users to easily visualize and check OWAMcap datasets.

## Public hosted

We offer a public hosted viewer at [https://huggingface.co/spaces/open-world-agents/visualize_dataset](https://huggingface.co/spaces/open-world-agents/visualize_dataset).

You can provide a huggingface repo id, or you can also upload your own OWAMcap dataset file via the viewer.
Note that this public hosted viewer has a 100MB upload file size limit. If you need to upload larger files, you may self-host the viewer.

## Self hosted

0. Go to `projects/owa-mcap-viewer` directory.
1. Setup `EXPORT_PATH` environment variable. You may setup `.env` or use `export` command.
    ```
    export EXPORT_PATH=(path-to-your-folder-containing-mcap-and-mkvs)
    ```
2. Run `vuv install` for installing dependencies.
3. Run the server with `uvicorn owa_viewer:app --host 0.0.0.0 --port 7860 --reload`
4. Access `http://localhost:7860` in your browser.