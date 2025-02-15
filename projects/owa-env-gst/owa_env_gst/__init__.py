import os

# set GST_PLUGIN_PATH to the 'gst-plugins' directory in the current working directory
os.environ["GST_PLUGIN_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gst-plugins")


from . import gst_factory


def activate():
    from . import screen  # noqa
    from . import recorder  # noqa
