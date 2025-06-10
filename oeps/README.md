# Open World Agents Enhancement Proposals (OEPs)

This directory contains Enhancement Proposals for Open World Agents (OWA), which serve as the primary mechanism for proposing, documenting, and tracking major changes to the framework.

## What are OEPs?

OEPs (Open World Agents Enhancement Proposals) are design documents that provide information to the OWA community or describe new features for the framework. Each OEP should provide a concise technical specification and rationale for the proposed feature.

## Quick Start

- **New to OEPs?** Start with [OEP-0000](oep-0000.md) for complete guidelines and process
- **Want to propose a feature?** Follow the workflow in OEP-0000 and use the provided template
- **Looking for existing proposals?** Browse the list below or check the [status summary](#status-summary)

## Current OEPs

| OEP | Title | Status | Type |
|-----|-------|--------|------|
| [0](oep-0000.md) | OEP Purpose and Guidelines | Active | Process |
| [1](oep-0001.md) | Core Component Design of OWA's Env - Callable, Listener, and Runnable | Final | Standards Track |
| [2](oep-0002.md) | Registry Pattern and Module System for OWA's Env | Final | Standards Track |

## Status Summary

- **Active**: 1 (OEP-0000)
- **Final**: 2 (OEP-0001, OEP-0002)
- **Draft**: 0
- **Total**: 3

## OEP Types

- **Standards Track**: New features or implementations for OWA
- **Informational**: Design issues, guidelines, or general information
- **Process**: Changes to OWA development processes or tools

## Key Design Principles

OWA's architecture is guided by several core principles documented in these OEPs:

- **Real-time Performance**: Sub-30ms latency for critical operations
- **Asynchronous Design**: Event-driven architecture with Callables, Listeners, and Runnables
- **Modular Plugin System**: Dynamic component registration and activation
- **Multimodal Data Handling**: Comprehensive desktop data capture and processing
- **Community-Driven**: Extensible framework with clear plugin interfaces

## Contributing

To propose a new OEP:

1. Review [OEP-0000](oep-0000.md) for complete guidelines
2. Discuss your idea with the community first
3. Fork the repository and create `oep-NNNN.md` using the next available number
4. Follow the template and format requirements in OEP-0000
5. Submit a pull request for review

## Implementation Status

- **OEP-1**: ✅ Fully implemented in `owa-core` package
- **OEP-2**: ✅ Fully implemented with registry system and module activation

## References

This OEP format is inspired by:
- [Python Enhancement Proposals (PEPs)](https://github.com/python/peps)
- [ROS Enhancement Proposals (REPs)](https://github.com/ros-infrastructure/rep)

## License

All OEPs are placed in the public domain or under the CC0-1.0-Universal license, whichever is more permissive.