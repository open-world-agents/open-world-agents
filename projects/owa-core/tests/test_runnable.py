import pytest

from owa.core.runnable import RunnableProcess, RunnableThread


class MyThreadTest(RunnableThread):
    def loop(self, *, stop_event):
        while not stop_event.is_set():
            stop_event.wait(1)


class MyProcessTest(RunnableProcess):
    def loop(self, *, stop_event):
        while not stop_event.is_set():
            stop_event.wait(1)


@pytest.mark.timeout(2)
def test_my_thread():
    """Test creation, start, and stop of a RunnableThread."""
    t = MyThreadTest().configure()
    t.start()
    # Wait a few seconds to confirm it is in the running state
    t.join(1)
    assert t.is_alive(), "Thread should be running."
    # Now stop it
    t.stop()
    t.join()
    assert not t.is_alive(), "Thread should have stopped."


@pytest.mark.timeout(2)
def test_my_process():
    """Test creation, start, and stop of a RunnableProcess."""
    p = MyProcessTest().configure()
    p.start()
    # Wait a few seconds to confirm it is in the running state
    p.join(1)
    assert p.is_alive(), "Process should be running."
    # Now stop it
    p.stop()
    p.join()
    assert not p.is_alive(), "Process should have stopped."
