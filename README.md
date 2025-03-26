# CodeWhisperer

<div align="center">
  <img src="logo.jpg" alt="AI Codebase Reviewer Logo" width="200">
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful AI-powered code review assistant that provides real-time feedback on your codebase. This tool leverages Together AI to detect critical issues, performance problems, and security vulnerabilities in your code.

## 🌟 Key Features

### 🔍 Code Review & Analysis
- **Live Code Review**: Real-time monitoring of code changes with immediate feedback
- **Critical Issue Detection**: Identifies bugs, performance issues, and security vulnerabilities
- **Static Analysis**: Built-in code analysis for common programming issues
- **Error Detection**: Syntax errors, undefined variables, security vulnerabilities, and logic flaws
- **Documentation Analysis**: Extract and analyze docstrings, comments, and code structure

### 💻 Project Management
- **Multi-Project Support**: Handle multiple codebases through an organized project system
- **Project Path Management**: Set and save default project paths
- **GitHub Integration**: Directly analyze repositories from GitHub
- **Configuration Management**: YAML-based settings and environment variable support

### 🚀 Processing & Performance
- **Parallel Processing**: Multi-threaded code analysis for better performance
- **Distributed Processing**: Support for distributed code analysis
- **Batch Processing**: Efficient handling of large codebases
- **Watch Mode**: Continuous monitoring of file changes

### 📊 Code Analysis
- **Structure Analysis**: Extract and analyze code structure (classes, functions, imports)
- **Similarity Scoring**: Calculate relevance scores for code suggestions
- **Multi-Language Support**: Reviews Python, JavaScript, TypeScript, Go, and more
- **Code Pattern Detection**: Identify and analyze common code patterns

### 🤖 AI Integration
- **Together AI Integration**: Powered by state-of-the-art language models
- **Smart Embeddings**: Generate and use embeddings for code analysis
- **Intelligent Suggestions**: Context-aware code completion and suggestions
- **Interactive Mode**: Ask questions and get explanations about your code

### 🛠️ Developer Tools
- **Rich CLI Interface**: Beautiful, interactive command-line interface with progress tracking
- **Debugging Support**: Error tracing and detailed error reporting
- **Progress Tracking**: Visual progress bars and status updates
- **Comprehensive Logging**: Detailed logging and error reporting system

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Together AI API key ([Get one here](https://www.together.ai/))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/boredom1234/CodeWhisperer.git
cd CodeWhisperer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your Together AI API key:
```bash
python cli.py configure
```

## 🎮 Usage

### Project Management

```bash
# Create a new project
python cli.py create-project myapp

# List available projects
python cli.py list-projects

# Switch to a different project
python cli.py switch-project myapp
```

### Code Review

```bash
# Set default project path
python cli.py assist --set-project-path /path/to/your/project

# Review using saved project path
python cli.py assist

# Review specific directory
python cli.py assist /path/to/review

# Watch mode for continuous review
python cli.py assist --watch
```

### Code Analysis

```bash
# Initialize codebase
python cli.py init /path/to/code --github

# Refresh codebase index
python cli.py refresh --summary

# Get code completions
python cli.py complete file.py --line 42

# Get inline suggestions
python cli.py suggest file.py --line 42

# Explain code sections
python cli.py explain file.py --start-line 10 --end-line 20
```

### Interactive Features

```bash
# Ask questions about code
python cli.py ask -i "How does X work?"

# Reset conversation history
python cli.py reset-history

# Show project status
python cli.py status --format json
```

### Review Output Format

The tool provides concise, focused feedback in the following format:
```
file.py:42 - Error: Description of the critical issue
file.py:73 - Warning: Description of the potential problem
```

Issues are categorized into:
- Critical bugs and errors
- Major performance issues
- Security vulnerabilities
- Significant design flaws

## 📚 Command Reference

| Command | Description | Options |
|---------|-------------|---------|
| `assist` | Start code review | `--watch`, `--set-project-path` |
| `ask` | Query the codebase | `--interactive`, `--composer`, `--chunks`, `--reset`, `--project` |
| `configure` | Set up API keys | - |
| `create-project` | Create/switch project | `PROJECT_NAME` |
| `init` | Initialize codebase | `--github`, `--project`, `--summary`, `--workers`, `--distributed` |
| `list-projects` | Show all projects | - |
| `switch-project` | Change active project | `PROJECT_NAME` |
| `status` | Show project statistics | `--format [text\|json]` |
| `refresh` | Update codebase index | `--summary` |
| `complete` | Get code completions | `--line`, `--scope`, `--patterns` |
| `suggest` | Get inline suggestions | `--line`, `--scope`, `--threshold` |
| `explain` | Explain code sections | `--start-line`, `--end-line`, `--detail` |
| `reset-history` | Clear conversation | - |
| `debug-projects` | Debug project registry | - |
| `compose` | Manage components | `add`, `remove`, `list` |

## 🛠️ Configuration

Configure through:

1. Environment variables (`.env`):
```env
TOGETHER_API_KEY=your_api_key_here
```

2. Project configuration (`~/.codeai/config.yml`):
```yaml
project_path: /path/to/your/project
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Together AI](https://www.together.ai/) for their powerful LLM API
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Rich](https://github.com/Textualize/rich) for beautiful terminal interfaces
- [Click](https://click.palletsprojects.com/) for CLI framework
