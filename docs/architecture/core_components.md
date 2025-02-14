# Core Components

The Open World Agents system is built on a set of core components that provide the foundation for modularity and dynamic functionality. These components work together to deliver a scalable and extensible framework:

- **Global Registry:** The system uses two key dictionaries, CALLABLES and LISTENERS, to store and manage functions and event handlers. This design promotes loose coupling between modules.
- **Module Activation:** Modules are loaded and activated at runtime using the `activate_module` function, allowing for dynamic extension of functionalities without impacting the core system.
- **Inter-module Communication:** Through the global registry, modules can communicate seamlessly, promoting reusability and decoupling.
- **Core Services:** Additional services such as logging, error handling, and configuration management support the operation and integration of various modules.

Each of these components is designed to ensure that new features and modules can be integrated with minimal friction, fostering innovation and rapid development.
