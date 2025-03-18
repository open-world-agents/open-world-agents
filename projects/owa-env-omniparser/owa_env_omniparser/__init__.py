"""
OmniParser integration for Open World Agents.

This module provides integration with Microsoft OmniParser,
enabling UI screenshot parsing capabilities within the Open World Agents framework.
"""


def activate():
    """
    Activate the OmniParser environment module.

    This function is called by the OWA framework when the module is activated.
    It imports and registers all callables and listeners provided by this module.
    """
    from . import callables  # noqa
    from . import api_client  # noqa
    from . import unified  # noqa
    from . import model_manager  # noqa
