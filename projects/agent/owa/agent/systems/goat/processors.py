from owa.agent.core.perception import Perception, PerceptionSpecDict, apply_spec


def perception_to_conversation(
    perception_history: Perception, current_perception: Perception, *, now: int, spec: PerceptionSpecDict
) -> tuple[Perception, dict]:
    """For events later than 'now', it's considered as future events('label')."""
    # TODO: implement sampling and trimming logic
    perception_history += current_perception
    perception_history, info = apply_spec(perception_history, now=now, spec=spec)
    if perception_history:
        items = "Summary of Perception History:\n"
        for channel, events in perception_history.items():
            # items.append(channel, len(events))
            items += f"{channel}: {len(events)} events\n"

        conversation = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": items},
                ],
            },
        ]
    else:
        conversation = None
    return perception_history, conversation


def lazy_load_images(inputs):
    # Placeholder for the actual image loading logic
    return inputs


def apply_processor(inputs, *, processor):
    # Placeholder for the actual processor application logic
    return {"input_ids": inputs[0]["content"][0]["text"]}
