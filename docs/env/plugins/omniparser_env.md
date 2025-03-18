# OmniParser Environment

The OmniParser Environment module (owa-env-omniparser) extends Open World Agents with computer vision capabilities to understand UI elements from screenshots. It integrates Microsoft's OmniParser technology, enabling agents to identify and interact with buttons, text fields, and other UI components.

## Features

- **UI Element Detection:** Parse screenshots to identify buttons, text boxes, dropdowns, and other UI elements.
- **Element Description Generation:** Generate natural language descriptions of detected UI elements.
- **Dual-Mode Operation:** Support for both embedded mode (local execution) and API server mode (remote execution).
- **Automatic Fallback:** If one mode fails, the system automatically attempts the other mode.
- **UI Element Search:** Helper functions to find specific UI elements by description.

## Usage

To activate the OmniParser Environment module, include the following in your code:

```python
from owa.registry import CALLABLES, activate_module

# Activate the OmniParser environment
activate_module("owa_env_omniparser")
```

After activation, you can access OmniParser functionalities via the global registries:

```python
# Capture screen and parse UI elements
screen_image = CALLABLES["screen.capture"]()
parsed_result = CALLABLES["screen.parse"](screen_image)

# Find and click on a specific element
CALLABLES["ui.find_and_click"]("Submit button")
```

## Implementation Details

The OmniParser environment provides a unified interface that handles both embedded and API modes:

1. **Embedded Mode:** Runs OmniParser locally with automatic model weight downloading.
2. **API Server Mode:** Connects to a remote OmniParser API server.
3. **Unified Interface:** Automatically manages both modes with fallback capabilities.

Configuration is managed via environment variables:

```
# API Server Configuration
OMNIPARSER_API_URL=https://your-api-server.com/api
OMNIPARSER_API_KEY=your-api-key

# Embedded Mode Configuration
OMNIPARSER_MODEL_PATH=/path/to/models
OMNIPARSER_EMBEDDED_ENABLED=true
```

To see detailed implementation, refer to the [owa-env-omniparser](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-omniparser) repository.

## Available Functions

### Screen Functions

- `screen.parse` - Parse a screenshot to identify UI elements
- `screen.describe` - Generate a description of all detected UI elements

### UI Interaction Functions

- `ui.find_element` - Find a specific UI element by description
- `ui.find_and_click` - Find a UI element by description and click on it
- `ui.get_element_bounds` - Get the bounding box coordinates of a UI element
- `ui.get_element_description` - Get the natural language description of a UI element

## Known Issues and Limitations

- Performance may vary depending on the complexity of the UI being parsed
- Some non-standard UI elements might not be correctly identified
- The embedded mode requires more memory and processing power than the API mode

## Future Development

Upcoming features for the OmniParser environment include:

1. Enhanced UI interaction helpers (drag-and-drop, form filling)
2. Improved element detection accuracy
3. Performance optimizations for faster parsing
4. Support for additional UI frameworks and platforms

(or just waiting for updates on origianl omniparser)

## Acknowledgments

This environment module is based on [Microsoft's OmniParser](https://github.com/microsoft/OmniParser), a powerful screen parsing tool designed for pure vision-based GUI agents. OmniParser is a comprehensive method for parsing user interface screenshots into structured and easy-to-understand elements, significantly enhancing the ability of vision models to generate accurately grounded actions.

The original OmniParser, developed by Microsoft, includes:

- Icon detection models for identifying UI elements in screenshots
- Caption generation models for describing detected elements
- A client-server architecture for flexible deployment

We also acknowledge [OmniTool](https://github.com/microsoft/OmniParser/tree/master/omnitool), which extends OmniParser to provide a complete solution for controlling Windows environments with AI-powered vision models. OmniTool demonstrates the practical applications of UI parsing for agents that can interact with computer interfaces.

Our integration brings the powerful capabilities of OmniParser to the Open World Agents framework while maintaining compatibility with the original models and adding features specific to our multi-agent architecture. We are grateful to the Microsoft team for their groundbreaking work in this area.

If you use this module in your research or applications, please consider citing the original OmniParser work:

```
@misc{lu2024omniparserpurevisionbased,
      title={OmniParser for Pure Vision Based GUI Agent},
      author={Yadong Lu and Jianwei Yang and Yelong Shen and Ahmed Awadallah},
      year={2024},
      eprint={2408.00203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00203},
}
```
