# Registry Pattern

The Open World Agents codebase uses a flexible registry pattern to manage module functionalities. This design allows modules to be dynamically registered without modifying the core system.

## Key Concepts

- **CALLABLES:** A global dictionary that stores references to synchronous functions provided by various modules (e.g., `clock.time_ns`).
- **LISTENERS:** A global dictionary that holds event listener classes for handling asynchronous events (e.g., the `clock/tick` listener).

## Module Activation

Modules are activated via the `activate_module` function. During activation, each module registers its functions and listeners to the global registries, making them available throughout the system.

## Benefits

- **Modularity:** Seamlessly integrate new functionalities.
- **Dynamic Integration:** Enable or disable modules at runtime.
- **Decoupling:** Keep the core system independent from specific module implementations.

Further details and advanced usage of the registry pattern will be expanded as the project evolves.
