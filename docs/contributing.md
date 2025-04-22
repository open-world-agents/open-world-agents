We'd love you to contribute to OWA!

## Issues

Questions, feature requests and bug reports are all welcome as [discussions or issues](https://github.com/open-world-agents/open-world-agents/issues/new/choose).
**However, to report a security vulnerability, please see our [security policy](https://github.com/open-world-agents/open-world-agents/security/policy).**
<!-- 
To make it as simple as possible for us to help you, please include the output of the following call in your issue:

```bash
python -c "import pydantic.version; print(pydantic.version.version_info())"
```
If you're using Pydantic prior to **v2.0** please use:
```bash
python -c "import pydantic.utils; print(pydantic.utils.version_info())"
```

Please try to always include the above unless you're unable to install Pydantic or **know** it's not relevant
to your question or feature request. -->

## Pull Requests

Feel free to create a Pull Request. Project maintainers will take a review quickly and give you a comments.

To make contributing as easy and fast as possible, you'll want to run tests and linting locally. Luckily,
OWA has few dependencies, doesn't require compiling and tests don't need access to databases, etc.
Because of this, setting up and running the tests should be very simple.

## Run tests

We're utilizing `pytest` for testing and `ruff` for formatting. Make sure your PR pass all tests in Github Actions.

1. Run `pytest`.
2. Run `ruff check`.
3. Run `ruff format`.

### How to Test Documentation Changes

If you contribute to the documentation—such as by adding new markdown files under `docs/data/`—please follow these steps to preview your changes locally before submitting a pull request:

1. **Add your documentation:**  
   Place your new markdown (`.md`) file in the `docs/data/` directory.

2. **Update navigation:**  
   Edit `mkdocs.yaml` to include your new file in the site's navigation (`nav` section).

3. **Install documentation dependencies:**  
   At the root of the project, run:  
   ```
   vuv install --extra docs
   ```
   This will install all necessary packages for building and serving the documentation site.

4. **Serve the documentation locally:**  
   Start the local documentation server with:  
   ```
   mkdocs serve
   ```

5. **Preview your changes:**  
   Open [http://localhost:8000](http://localhost:8000) in your web browser to view the documentation and verify your additions appear as expected.

Following these steps will help ensure your documentation contributions are correctly displayed and formatted. Thank you for helping improve the OWA documentation!