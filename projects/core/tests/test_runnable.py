import pytest

from owa.runnable import RunnableProcess, RunnableThread


class MyThreadTest(RunnableThread):
    def loop(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(1)

    def cleanup(self): ...


class MyProcessTest(RunnableProcess):
    def loop(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(1)

    def cleanup(self): ...


@pytest.mark.timeout(3)
def test_my_thread():
    """Test creation, start, and stop of a RunnableThread."""
    t = MyThreadTest()
    t.start()
    # Wait a few seconds to confirm it is in the running state
    t.join(2)
    assert t.is_alive(), "Thread should be running."
    # Now stop it
    t.stop()
    t.join()
    assert not t.is_alive(), "Thread should have stopped."


@pytest.mark.timeout(3)
def test_my_process():
    """Test creation, start, and stop of a RunnableProcess."""
    p = MyProcessTest()
    p.start()
    # Wait a few seconds to confirm it is in the running state
    p.join(2)
    assert p.is_alive(), "Process should be running."
    # Now stop it
    p.stop()
    p.join()
    assert not p.is_alive(), "Process should have stopped."
