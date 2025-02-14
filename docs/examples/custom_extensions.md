# Custom Extensions

This document provides a brief overview of how to implement custom module extensions in Open World Agents. Custom extensions allow you to integrate domain-specific functionalities seamlessly into the system.

## Example: Minecraft Integration

The following example demonstrates how to activate a custom module that integrates with Minecraft. This module provides functionalities to interact with in-game elements:

```python
# Activate the custom Minecraft module
activate_module("owa_minecraft")

# Retrieve the current inventory of a player (e.g., "Steve")
inventory = CALLABLES["minecraft.get_inventory"](player="Steve")
print(inventory)
```

## Guidelines for Developing Custom Extensions

- Ensure that your custom module adheres to the registry pattern by registering its functions and listeners to the appropriate global registries.
- Avoid modifying core functionalities directly; leverage the extendable structure provided by Open World Agents.
- Write comprehensive tests to ensure compatibility and stability within the broader system.
- Update documentation as you add new features or extend existing ones.

Custom extensions empower you to tailor the system to your specific needs while benefiting from the core strengths of Open World Agents.
