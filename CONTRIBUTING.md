# Contributing to CodeWhisperer

Thank you for your interest in contributing to CodeWhisperer! This document provides guidelines and instructions for contributing to this project. By participating, you are expected to uphold this code and help us create a positive and collaborative environment.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Contact](#contact)

## Code of Conduct

### Our Pledge

We pledge to make participation in our project and community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Unacceptable behavior includes:
- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

### Enforcement

Project maintainers are responsible for clarifying the standards of acceptable behavior and will take appropriate and fair corrective action in response to any instances of unacceptable behavior.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- pip package manager
- A Together AI API key (for testing)

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/CodeWhisperer.git
   cd CodeWhisperer
   ```

3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If applicable
   ```

5. Set up pre-commit hooks:
   ```bash
   pre-commit install  # If pre-commit is used
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-you-are-fixing
   ```

2. Make your changes, following the [coding standards](#coding-standards)

3. Run tests to ensure your changes don't break existing functionality:
   ```bash
   pytest
   ```

4. Commit your changes:
   ```bash
   git commit -m "Your detailed commit message"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request

## Pull Request Process

1. Update the README.md or relevant documentation with details of changes if applicable
2. Ensure any new code is well-tested and passes all tests
3. Update the version numbers in any example files and the README.md to the new version that the Pull Request would represent
4. The Pull Request will be merged once it receives approval from at least one project maintainer

## Coding Standards

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code. Additionally:

- Use the [Black](https://github.com/psf/black) code formatter
- Sort imports using [isort](https://pycqa.github.io/isort/)
- Use type hints where appropriate
- Write docstrings in the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep lines to a maximum of 88 characters
- Use meaningful variable and function names

### Example Code Style

```python
from typing import List, Optional

def process_data(data: List[str], config: Optional[dict] = None) -> dict:
    """Process the input data according to the configuration.

    Args:
        data: A list of strings to process.
        config: Optional configuration dictionary.

    Returns:
        A dictionary containing the processed results.
    """
    if not data:
        return {"error": "No data provided"}

    if config is None:
        config = {"default_mode": True}

    # Process data here
    results = {"processed": True, "items": len(data)}
    return results
```

## Testing Guidelines

- Write unit tests for all new code
- Aim for test coverage of at least 80%
- Use pytest for writing and running tests
- Mock external services where appropriate
- Include both positive and negative test cases

### Example Test

```python
import pytest
from your_module import process_data

def test_process_data_with_valid_input():
    data = ["item1", "item2"]
    result = process_data(data)
    assert result["processed"] is True
    assert result["items"] == 2

def test_process_data_with_empty_input():
    result = process_data([])
    assert "error" in result
    assert result["error"] == "No data provided"
```

## Documentation

Good documentation is essential for the project's success. Please follow these guidelines:

- Update the README.md with any changes to the interface or functionality
- Document all public functions, classes, and methods with docstrings
- Keep docstrings up-to-date when modifying code
- Add inline comments for complex logic
- For significant changes or additions, update the wiki (if applicable)

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- A clear and descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Screenshots if applicable
- Your environment (OS, Python version, etc.)
- Any additional context

### Feature Requests

For feature requests, include:

- A clear and descriptive title
- A detailed description of the proposed feature
- Any relevant examples or use cases
- Possible implementation approaches (if you have ideas)
- Why this feature would be beneficial to the project

## Contact

If you have questions or need help, you can:

- Open an issue with your question
- Join our community Discord/Slack (if applicable)
- Contact the project maintainers directly (provide contact information if appropriate)

---

Thank you for contributing to CodeWhisperer! Your efforts help make this project better for everyone. 