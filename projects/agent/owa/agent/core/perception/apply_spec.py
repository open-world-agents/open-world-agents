from owa.core.time import TimeUnits

from .perception import Perception
from .spec import PerceptionSpec, PerceptionSpecDict


def apply_spec(perception: Perception, *, now: int, spec: PerceptionSpecDict) -> tuple[Perception, dict]:
    """
    Apply the perception spec to the given perception.

    Args:
        perception (Perception): The perception to apply the spec to.
        spec (PerceptionSpecDict): The specification to apply.

    Returns:
        tuple[Perception, dict]: The modified perception and a dictionary of additional information.
    """

    result = Perception()
    info = {}

    for topic in perception.keys():
        events = perception[topic]
        topic_spec: PerceptionSpec = spec[topic]
        original_count = len(events)

        # Filter events within the window, using now as reference
        window_start = now + topic_spec.window_start * TimeUnits.SECOND
        window_end = now + topic_spec.window_end * TimeUnits.SECOND
        events_in_window = [e for e in events if window_start <= e.timestamp <= window_end]

        # Apply sampling configs
        for sampling in topic_spec.sample_configs:
            sampled = []
            target_events = []
            non_target_events = []
            for e in events_in_window:
                if sampling.sample_if is None or sampling.sample_if(e):
                    target_events.append(e)
                else:
                    non_target_events.append(e)
            # Downsample if sampling_rate is set (naive implementation: uniform step)
            if sampling.sampling_rate > 0 and len(target_events) > 1:
                if sampling.do_interpolate:
                    raise NotImplementedError("Interpolation is not implemented yet.")

                # Sort by timestamp
                target_events.sort(key=lambda e: e.timestamp)
                interval = 1.0 * TimeUnits.SECOND / sampling.sampling_rate
                last_time = None
                for ev in target_events:
                    if last_time is None or ev.timestamp - last_time >= interval:
                        sampled.append(ev)
                        last_time = ev.timestamp

            # Merge back with non-target events
            events_in_window = sorted(sampled + non_target_events, key=lambda e: e.timestamp)

        # Apply trim configs
        for trim in topic_spec.trim_configs:
            trimmed = []
            trim_targets = []
            non_trim_targets = []
            for e in events_in_window:
                if trim.trim_if is None or trim.trim_if(e):
                    trim_targets.append(e)
                else:
                    non_trim_targets.append(e)

            if trim.trim_mode == "first_k" and trim.trim_k is not None:
                trimmed = trim_targets[: trim.trim_k]
            elif trim.trim_mode == "last_k" and trim.trim_k is not None:
                trimmed = trim_targets[-trim.trim_k :]
            else:
                trimmed = trim_targets

            # Merge back with non-target events
            events_in_window = sorted(trimmed + non_trim_targets, key=lambda e: e.timestamp)

        result[topic] = events_in_window
        info[topic] = {
            "original_count": original_count,
            "window_count": len(events_in_window),
        }

    return result, info
