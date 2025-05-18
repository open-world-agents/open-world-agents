from queue import Queue


def iter_queue(queue: Queue):
    while not queue.empty():
        yield queue.get_nowait()


def get_last_from_queue(queue: Queue, *args, **kwargs):
    item = queue.get(*args, **kwargs)
    while not queue.empty():
        item = queue.get_nowait()
    return item
