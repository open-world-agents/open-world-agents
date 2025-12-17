"""
Element that generates UTC timestamps in text/x-raw,format=utf8 for subtitle tracks.
Reference: https://github.com/GStreamer/gst-python/blob/master/examples/plugins/python/py_audiotestsrc.py

Example pipeline:

gst-launch-1.0 utctimestampsrc ! fakesink dump=1
gst-launch-1.0 -e -v videotestsrc is-live=true ! x264enc ! h264parse ! queue ! mux. `
    utctimestampsrc interval=1 ! queue ! mux. `
    matroskamux name=mux ! filesink location=output_with_subtitles.mkv
"""

# ruff: noqa: F841, F401
# ruff: noqa: E402
# To suppress the warning for E402, waiting for https://github.com/astral-sh/ruff/issues/3711
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")

import datetime
import struct
import sys
import time

from gi.repository import GObject, Gst, GstBase
from loguru import logger

# Initialize GObject and Gst

if not Gst.is_initialized():
    Gst.init(None)

OCAPS = Gst.Caps.from_string("text/x-raw,format=utf8")

DEFAULT_INTERVAL = 1  # in seconds

# logger.remove()
# logger.add(sys.stderr, level="TRACE")


# This can't be changed to PushSrc because of basic issue: https://gitlab.freedesktop.org/gstreamer/gst-python/-/issues/1
class UtcTimestampSrc(GstBase.BaseSrc):
    __gstmetadata__ = (
        "UtcTimestampSrc",
        "Source",
        "Source element that outputs UTC timestamps in SRT format",
        "MilkClouds",
    )

    __gsttemplates__ = Gst.PadTemplate.new("src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, OCAPS)

    __gproperties__ = {
        "interval": (
            float,
            "Interval",
            "Time interval between outputting timestamps (in seconds)",
            1e-3,
            3600,
            DEFAULT_INTERVAL,
            GObject.ParamFlags.READWRITE,
        )
    }

    def __init__(self):
        super(UtcTimestampSrc, self).__init__()
        self.interval = DEFAULT_INTERVAL
        self.set_live(True)
        self.set_format(Gst.Format.TIME)
        self.past_time = None

    def do_get_property(self, prop):
        if prop.name == "interval":
            return self.interval
        else:
            raise AttributeError("Unknown property %s" % prop.name)

    def do_set_property(self, prop, value):
        if prop.name == "interval":
            self.interval = value
        else:
            raise AttributeError("Unknown property %s" % prop.name)

    def do_start(self):
        self.past_time = None
        return True

    # deprecated
    def wait_next(self, buf):
        clock = self.get_clock()
        if not clock:
            Gst.error("Clock is not available.")
            return Gst.FlowReturn.ERROR

        if self.next_time is None:
            self.next_time = clock.get_time() - self.get_base_time() + buf.duration

        abs_time = self.next_time + self.get_base_time()
        clock_id = clock.new_single_shot_id(abs_time)
        # https://gstreamer.freedesktop.org/documentation/gstreamer/gstclock.html?gi-language=python#gst_clock_id_wait
        ret, jitter = clock.id_wait(clock_id)

        if ret == Gst.ClockReturn.UNSCHEDULED:
            clock.id_unref(clock_id)
            return Gst.FlowReturn.FLUSHING
        elif ret != Gst.ClockReturn.OK:
            clock.id_unref(clock_id)
            Gst.error("Clock wait error: %s" % ret)
            return Gst.FlowReturn.ERROR

        clock.id_unref(clock_id)

    def do_fill(self, offset, length, buf):
        if self.past_time is None:
            self.past_time = time.time_ns()

        to_sleep = max(0, self.interval - (time.time_ns() - self.past_time) / 1e9)
        time.sleep(to_sleep)
        self.past_time = self.past_time + self.interval * 1e9

        # Set buffer duration
        buf.duration = int(self.interval * Gst.SECOND)
        # Get the current UTC time in nanoseconds
        current_time = time.time_ns()
        utc_time = datetime.datetime.fromtimestamp(current_time / 1e9).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        # Set buffer PTS
        pts_time = self.get_clock().get_time() - self.get_base_time()
        buf.pts = pts_time

        logger.trace(
            (
                (self.get_clock().get_time() - self.get_base_time()) / Gst.SECOND,
                buf.pts / Gst.SECOND,
                buf.duration / Gst.SECOND,
            )
        )

        # Output plain text for text/x-raw,format=utf8
        # The content is just the UTC timestamp in nanoseconds
        data = f"{current_time}".encode("utf-8")

        buf.set_size(len(data))

        # Map the buffer to write data
        try:
            with buf.map(Gst.MapFlags.WRITE) as map_info:
                # Fill the buffer with the subtitle data
                map_info.data[: len(data)] = data
        except Exception as e:
            Gst.error("Mapping error: %s" % e)
            return Gst.FlowReturn.ERROR

        return (Gst.FlowReturn.OK, buf)


__gstelementfactory__ = ("utctimestampsrc", Gst.Rank.NONE, UtcTimestampSrc)
