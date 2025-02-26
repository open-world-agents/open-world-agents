# ruff: noqa: E402
# To suppress the warning for E402, waiting for https://github.com/astral-sh/ruff/issues/3711
import gi

gi.require_version("Gst", "1.0")

from gi.repository import GLib, Gst
from loguru import logger

from owa import Runnable

from ..utils import try_set_state

# Initialize GStreamer
if not Gst.is_initialized():
    Gst.init(None)


def on_message(bus: Gst.Bus, message: Gst.Message, loop: GLib.MainLoop):
    """
    Handle GStreamer bus messages.

    Args:
        bus: GStreamer bus object
        message: Message received from the bus
        loop: GLib main loop
    """
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        logger.info("Received EOS signal, shutting down gracefully.")
        loop.quit()
    elif msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error(f"Error received: {err}, {debug}")
        loop.quit()


class BaseGstPipelineRunner(Runnable):
    """
    A generalized GStreamer pipeline runner that manages pipeline lifecycle and callbacks.
    """

    def on_configure(self, pipeline_description: str) -> bool:
        """
        Configure the GStreamer pipeline.

        Args:
            pipeline_description: GStreamer pipeline description string
            start_time: Starting time position in seconds (optional)

        Returns:
            bool: Configuration success status
        """
        self.pipeline_description = pipeline_description
        self.pipeline = None
        self._loop = None
        self.appsinks = []

        try:
            self.pipeline = Gst.parse_launch(self.pipeline_description)
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            return False

        if not self.pipeline:
            logger.error("Failed to create pipeline from description.")
            return False

        self._loop = GLib.MainLoop()

        # Setup bus message handling
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", on_message, self._loop)

        return True

    def loop(self):
        """Run the main GLib loop."""
        try_set_state(self.pipeline, Gst.State.PLAYING)
        self._loop.run()

    def cleanup(self):
        """Clean up pipeline resources."""
        if self._loop:
            self._loop.quit()
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        self.pipeline = None
        self.appsinks = []
        self._loop = None

    def stop(self):
        """Stop the pipeline gracefully."""
        if self.pipeline:
            self.pipeline.send_event(Gst.Event.new_eos())
            # After sending EOS, `on_message` will handle the EOS signal and quit the loop

    def find_elements_by_name(self, name: str) -> list[Gst.Element]:
        """
        Find an element by name in the pipeline.

        Args:
            name: Name of the element to find

        Returns:
            list[Gst.Element]: List of elements found
        """
        elements = []
        try:
            iter = self.pipeline.iterate_sinks()
            while True:
                res, elem = iter.next()
                if res == Gst.IteratorResult.OK:
                    if isinstance(elem, Gst.Element) and elem.get_factory().get_name() == name:
                        elements.append(elem)
                elif res == Gst.IteratorResult.DONE:
                    break
                elif res == Gst.IteratorResult.ERROR:
                    raise Exception("Error iterating over sink elements")
                elif res == Gst.IteratorResult.RESUME:
                    continue
        except Exception as e:
            raise Exception(f"Error while iterating sink elements: {e}")

        if not elements:
            logger.warning(f"No {name} found in pipeline.")

        return elements
