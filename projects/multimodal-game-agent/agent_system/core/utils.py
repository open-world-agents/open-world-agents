def dequeue_perceptions(perception_queue):
    current_perceptions = []
    try:
        while True:
            p = perception_queue.get_nowait()
            current_perceptions.append(p)
    except Exception:
        pass  # Or queue.Empty
    return current_perceptions


def dequeue_decision(decision_queue):
    return decision_queue.get() if not decision_queue.empty() else None


enqueue_thought = lambda x: None  # Placeholder
enqueue_action = lambda: None  # Placeholder


def decision_to_action(decision):
    pass
