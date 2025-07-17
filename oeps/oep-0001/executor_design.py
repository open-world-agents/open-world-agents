import time
from typing import Callable

task = runner.execute(configure_fn, loop_fn, cleanup_fn)

task.start()
time.sleep(3)
task.stop()
task.join()

with task.context:
    time.sleep(3)

"""
executor가 필요하지만 executor의 역할을 외부로 빼둘 수 있는 디자인
vs.
executor가 없지만 executor의 역할이 클래스에 들어가 있는 디자인


"""


class TaskDefinition:
    configure_fn: Callable
    loop_fn: Callable
    cleanup_fn: Callable

class ListenerDefinition:
    configure_fn: Callable
    loop_fn: Callable
    cleanup_fn: Callable
    event_cls: EventCls


class ExampleTask:
    def configure(self): ...
    def loop(self): ...
    def cleanup(self): ...


task = Task.from_class(ExampleTask)

class Task

class Executor:
    def