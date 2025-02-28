
To see detailed implementation, skim over [owa_env_desktop](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-desktop/owa_env_desktop). API Docs is being written WIP.


Below is just a list of callables:

- `mouse.click`
- `mouse.move`
- `mouse.position`
- `mouse.press`
- `mouse.release`
- `mouse.scroll`

- `keyboard.press`
- `keyboard.release`
- `keyboard.type`

- `screen.capture`: this module utilize `bettercam`. For better performance and extensibility, use `owa-env-gst`'s functions instead.

- `window.get_active_window`
- `window.get_window_by_title`
- `window.when_active`

Below is just a list of listeners:

- `keyboard`
- `mouse`