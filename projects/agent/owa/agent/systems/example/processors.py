from owa.agent.core import Event, Perception, PerceptionSpecDict


def perception_to_conversation(
    perception_history: Perception, current_perception: Perception, *, now, spec: PerceptionSpecDict
) -> tuple[Perception, dict]:
    """For events later than 'now', it's considered as future events('label')."""
    # Placeholder for the actual conversion logic
    if current_perception:
        conversation = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": [*current_perception.values()][0][0].msg},
                ],
            },
        ]
    else:
        conversation = None
    perception_history.extend(current_perception)
    # Leave only the latest perception in the history
    perception_history = perception_history[-1:]
    return perception_history, conversation


def lazy_load_images(inputs):
    # Placeholder for the actual image loading logic
    return inputs


def apply_processor(inputs, *, processor):
    # Placeholder for the actual processor application logic
    return {"input_ids": inputs[0]["content"][0]["text"]}
