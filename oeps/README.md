# Open World Agents Enhancement Proposals (OEPs)

This directory contains Enhancement Proposals for Open World Agents (OWA), which serve as the primary mechanism for proposing, documenting, and tracking major changes to the framework.

## What are OEPs?

OEPs (Open World Agents Enhancement Proposals) are design documents that provide information to the OWA community or describe new features for the framework. Each OEP should provide a concise technical specification and rationale for the proposed feature.

## Quick Start

- **New to OEPs?** Start with [OEP-0000](oep-0000.md) for complete guidelines and process
- **Want to propose a feature?** Follow the workflow in OEP-0000 and use the provided template
- **Looking for existing proposals?** Browse the list below or check the [status summary](#status-summary)

## Current OEPs

| OEP | Title | Status | Type | Summary |
|-----|-------|--------|------|---------|
| [0](oep-0000.md) | OEP Purpose and Guidelines | Active | Process | Defines the OEP process and template structure |
| [1](oep-0001.md) | Core Component Design of OWA's Env - Callable, Listener, and Runnable | Final | Standards Track | Establishes the three fundamental component types for the OWA environment system |
| [2](oep-0002.md) | Registry Pattern and Module System for OWA's Env | Superseded | Standards Track | Original registry system with manual activation (superseded by OEP-3) |
| [3](oep-0003.md) | Entry Points-Based Plugin Discovery and Unified Component Naming | Final | Standards Track | Modern plugin system with automatic discovery and unified naming conventions |
| [4](oep-0004.md) | Documentation Validation and mkdocstrings Integration for EnvPlugins | Final | Standards Track | Documentation quality assurance tools and mkdocstrings integration |
| [5](oep-0005.md) | Message Definition Registration in EnvPlugin System | Rejected | Standards Track | Plugin-based message registration (rejected in favor of OEP-6) |
| [6](oep-0006.md) | Dedicated OWA Message Package and OWAMcap Profile Specification | Final | Standards Track | Dedicated message package with entry point discovery and OWAMcap specification |

## Status Summary

- **Active**: 1 (OEP-0000)
- **Final**: 4 (OEP-0001, OEP-0003, OEP-0004, OEP-0006)
- **Superseded**: 1 (OEP-0002)
- **Rejected**: 1 (OEP-0005)
- **Draft**: 0
- **Total**: 7

## OEP Types

- **Standards Track**: New features or implementations for OWA
- **Informational**: Design issues, guidelines, or general information
- **Process**: Changes to OWA development processes or tools

## OEP Status Definitions

- **Draft**: The OEP is being actively worked on and is not yet ready for review
- **Accepted**: The OEP has been accepted and is being implemented
- **Final**: The OEP has been implemented and is considered complete
- **Rejected**: The OEP has been rejected and will not be implemented
- **Superseded**: The OEP has been replaced by a newer OEP
- **Active**: The OEP is an ongoing process (typically for Process type OEPs)

## Key Design Principles

OWA's architecture is guided by several core principles documented in these OEPs:

- **Real-time Performance**: Sub-30ms latency for critical operations (OEP-1)
- **Asynchronous Design**: Event-driven architecture with Callables, Listeners, and Runnables (OEP-1)
- **Modular Plugin System**: Entry points-based automatic discovery and unified naming (OEP-3)
- **Multimodal Data Handling**: Comprehensive desktop data capture and processing (OEP-1, OEP-6)
- **Community-Driven**: Extensible framework with clear plugin interfaces (OEP-3)
- **Documentation Quality**: Automated validation and quality assurance tools (OEP-4)
- **Message Standards**: Dedicated message package with standardized schemas (OEP-6)

## Contributing

To propose a new OEP:

1. Review [OEP-0000](oep-0000.md) for complete guidelines
2. Discuss your idea with the community first
3. Fork the repository and create `oep-NNNN.md` using the next available number
4. Follow the template and format requirements in OEP-0000
5. Submit a pull request for review

## Implementation Status

- **OEP-1**: ✅ Fully implemented in `owa-core` package
- **OEP-2**: ⚠️ Superseded by OEP-3 (legacy implementation available)
- **OEP-3**: ✅ Fully implemented with entry points-based plugin discovery
- **OEP-4**: ✅ Fully implemented with documentation validation and mkdocstrings integration
- **OEP-5**: ❌ Rejected in favor of OEP-6 approach
- **OEP-6**: ✅ Fully implemented with dedicated `owa-msgs` package and message registry

## References

This OEP format is inspired by:
- [Python Enhancement Proposals (PEPs)](https://github.com/python/peps)
- [ROS Enhancement Proposals (REPs)](https://github.com/ros-infrastructure/rep)

## License

All OEPs are placed in the public domain or under the CC0-1.0-Universal license, whichever is more permissive.