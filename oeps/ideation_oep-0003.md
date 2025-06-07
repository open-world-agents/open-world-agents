새로운 환경 “example”을 개발하고 등록하고 싶은 개발자 Bob가 있다고 하자.

이때, Bob은 다음 src를 담은 패키지를 개발한다.
```
owa/
└── env/
    ├── example.py
    └── example/
        └── __init__.py
pyproject.toml # indicates the package owa-env-desktop has src in `owa`
```

다음은 example.py의 내용물이다.
```py
from owa.core.env_plugin import PluginSpec
plugin_spec = PluginSpec({
    "namespace": "example",  # it is "alias" of EnvPlugin
    "version": "0.1.0",
    "description": "Example environment plugin for Open World Agents",
    "author": "OWA Team",
    "components": {
        "callables": {
            "callable": "owa.env.example.example_callable:ExampleCallable",
            "print": "owa.env.example.example_callable:example_print",
            "add": "owa.env.example.example_callable:example_add"
        },
        "listeners": {
            "listener": "owa.env.example.example_listener:ExampleListener",
            "timer": "owa.env.example.example_listener:ExampleTimer"
        },
        "runnables": {
            "runnable": "owa.env.example.example_runnable:ExampleRunnable",
            "counter": "owa.env.example.example_runnable:ExampleCounter"
        },
        "messages": {
            "event": "owa.env.example.example_message:ExampleEvent"
        },
    },
})
```


패키지의 사용자는 `pip install owa-env-example`로 설치한다.
이후 `owa.core.registry`에서는 다음 로직으로 설치된 패키지를 감지하여, plugin_spec을 등록한다.


```python
import importlib
import pkgutil

namespace = "owa.env"
ns = importlib.import_module(namespace)

for _, name, ispkg in pkgutil.iter_modules(ns.__path__):
    if not ispkg:
        print(f"{namespace}.{name}")
        # you can find example.py with this, and you may import `plugin_spec` from the file.
```

등록 결과 namespace "example"에 대한 callable, listener, runnable, messages가 등록된다.
이렇게 등록한 것들은 추후 다음과 같이 사용할 수 있다.

```
print = CALLABLES.get(namespace="example", name="print") 
print("Hello, World!") 
```

또한 다음과 같은 커맨드로 등록된 EnvPlugin과 그 내용물을 볼 수 있게 하자.
```sh
$ owl env list # list up EnvPlugins
$ owl env show example # outputs spec of EnvPlugin

# or, list up callable
$ owl env list callables
$ owl env list runnables
# ...
```
