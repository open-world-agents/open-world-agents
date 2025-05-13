from queue import Queue


def iter_queue(queue: Queue):
    while not queue.empty():
        yield queue.get_nowait()
