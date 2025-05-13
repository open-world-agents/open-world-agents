from owa.agent.core.spec import PerceptionSamplingSpec, SamplingStrategy

PERCEPTION_SAMPLING_SPEC = PerceptionSamplingSpec(
    strategies=[
        SamplingStrategy(-0.25, 0, 0.05, "screen"),
        SamplingStrategy(-0.25, 0, 0.05, "mouse"),
        SamplingStrategy(-0.25, 0, 0, "keyboard"),
        SamplingStrategy(0, 1, 0, "keyboard"),  # label
        SamplingStrategy(0, 1, 0.05, "mouse"),  # label
    ]
)
