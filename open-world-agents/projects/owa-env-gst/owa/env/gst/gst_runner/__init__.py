from .extensions import AppsinkExtension, FPSDisplayExtension, SeekExtension
from .gst_runner import BaseGstPipelineRunner


class GstPipelineRunner(BaseGstPipelineRunner, AppsinkExtension, FPSDisplayExtension, SeekExtension):
    """
    Example:
    ```python
    # Create and configure the pipeline runner
    runner = GstPipelineRunner().configure(pipeline_description)

    # AppsinkExtension: you may register appsink callback if appsink is included in pipeline
    runner.register_appsink_callback(sample_callback)

    # FPSDisplayExtension: you may enable fps display if fpsdisplaysink is included in pipeline
    runner.enable_fps_display()

    # SeekExtension: you may seek to a specific position before starting
    runner.seek(start_time=0.4, end_time=1.5)

    try:
        # Start the pipeline
        runner.start()

        # Monitor the pipeline
        while runner.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping pipeline.")
        runner.stop()
    finally:
        runner.join()
        logger.info("Pipeline stopped.")
    ```
    """
