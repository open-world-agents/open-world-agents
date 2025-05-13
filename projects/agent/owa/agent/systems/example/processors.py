from owa.agent.core import PerceptionSamplingSpec


def perception_to_conversation(
    perception_history, thought_history, current_perception, *, now, spec: PerceptionSamplingSpec
):
    """For events later than 'now', it's considered as future events('label')."""
    pass


def lazy_load_images(inputs):
    pass


def apply_processor(processor, inputs):
    pass
