# CodeCraft

<div align="center">
  <img src="logo.jpg" alt="AI Codebase Reviewer Logo" width="200">
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful AI-powered code review assistant that provides real-time feedback on your codebase. This tool leverages Together AI to detect critical issues, performance problems, and security vulnerabilities in your code.

## 📋 Table of Contents
- [Getting Started](#-getting-started)
- [Key Features](#-key-features)
- [Basic Workflow](#-basic-workflow)
- [Command Reference](#-command-reference)
- [Configuration](#-configuration)
- [Advanced Features](#-advanced-features)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Together AI API key ([Get one here](https://www.together.ai/))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/boredom1234/CodeCraft.git
cd CodeCraft
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

### Initial Setup

1. Create a new project:
```bash
python cli.py create-project myapp
```

2. Initialize your codebase:
```bash
python cli.py init /path/to/code
# Or initialize from GitHub
python cli.py init --github https://github.com/username/repo
```

3. Set your project path:
```bash
python cli.py assist --set-project-path /path/to/your/project
```

## 🌟 Key Features

### 🔍 Code Review & Analysis
- **Live Code Review**: Real-time monitoring of code changes with immediate feedback
- **Critical Issue Detection**: Identifies bugs, performance issues, and security vulnerabilities
- **Error Detection**: Syntax errors, undefined variables, security vulnerabilities, and logic flaws
- **Documentation Analysis**: Extract and analyze docstrings, comments, and code structure

### 💻 Project Management
- **Multi-Project Support**: Handle multiple codebases through an organized project system
- **GitHub Integration**: Directly analyze repositories from GitHub
- **Configuration Management**: YAML-based settings and environment variable support

### 🚀 Processing & Performance
- **Parallel Processing**: Multi-threaded code analysis for better performance
- **Memory-Aware Processing**: Adaptive batch sizing based on system memory
- **Disk Streaming**: Stream results to disk to avoid memory issues
- **Watch Mode**: Continuous monitoring of file changes
- **Batch Processing**: Efficient handling of large codebases
- **Distributed Processing**: Support for distributed workers

### 🤖 AI Integration
- **Together AI Integration**: Powered by state-of-the-art language models
- **Smart Embeddings**: Generate and use embeddings for code analysis
- **Response Formatting**: Format output with headers and footers for better readability
- **Chunked Responses**: Split long responses into chunks to avoid console truncation
- **Interactive Mode**: Ask questions and get explanations about your code
- **Concise Mode**: Get brief, direct answers instead of detailed explanations
- **Code-Aware Retrieval**: Advanced search that understands code structure and relationships

### 🔎 Advanced Code Search
- **Exact Line Search**: Find precisely where specific code lines are located
- **Cross-File References**: Understand relationships between components across files
- **Semantic Code Understanding**: Find code based on functionality, not just text matching
- **Context-Aware Results**: Results include necessary context from imports and related functions

## 💡 Basic Workflow

### 1. Initialize Your Project (Required First Step)
```bash
# Create a project and initialize codebase
python cli.py create-project myapp
python cli.py init /path/to/code
```

### 2. Perform Code Reviews
```bash
# Review your code
python cli.py assist

# Watch mode for continuous review
python cli.py assist --watch
```

### 3. Ask Questions About Your Code
```bash
# Ask a single question
python cli.py ask "How does the authentication system work?"

# Start an interactive session
python cli.py ask -i

# Get concise answers (1-2 sentences)
python cli.py ask --concise "What does the process_file function do?"

# Find exact code lines
python cli.py ask "Where is this line located: const user = useUser();"
```

### 4. Analyze Specific Code Sections
```bash
# Explain a section of code
python cli.py explain file.py --start-line 10 --end-line 20

# Get suggestions for specific lines
python cli.py suggest file.py --line 42
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
| `init` | **Initialize codebase (first step)** | `--github`, `--project`, `--summary`, `--workers`, `--batch-size`, `--distributed`, `--worker-addresses` |
| `create-project` | Create new project | `PROJECT_NAME` |
| `configure` | Set up API keys | - |
| `assist` | Start code review | `--watch`, `--set-project-path` |
| `ask` | Query the codebase | `--interactive`, `--composer`, `--chunks`, `--concise`, `--reset`, `--project`, `--model` |
| `explain` | Explain code sections | `--start-line`, `--end-line`, `--detail` |
| `suggest` | Get inline suggestions | `--line`, `--scope`, `--threshold` |
| `complete` | Get code completions | `--line`, `--scope`, `--patterns` |
| `list-projects` | Show all projects | - |
| `switch-project` | Change active project | `PROJECT_NAME` |
| `status` | Show project statistics | `--format [text\|json]` |
| `refresh` | Update codebase index | `--project`, `--summary` |
| `reset-history` | Clear conversation | - |
| `debug-projects` | Debug project registry | - |
| `compose` | Manage components | `add`, `remove`, `list` |
| `set-model` | Set default model | `--list` |

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

3. Response verbosity settings (in `config.yml`):
```yaml
# Set concise mode globally
response_mode: concise

# Or with more detailed settings
model:
  concise_responses: true
  max_tokens: 150
  verbosity: low
  default: meta-llama/Llama-3.3-70B-Instruct-Turbo
```

## 🧪 Advanced Features

### Memory-Aware Processing

The tool dynamically adjusts batch sizes based on available system memory:

```yaml
# In config.yml
parallel_processing:
  memory_limit_percentage: 80.0
  stream_to_disk: true
  cache_dir: .cache
```

### Distributed Processing

Enable distributed processing for larger codebases:

```bash
python cli.py init /path/to/code --distributed --worker-addresses host1:5000,host2:5000
```

### Semantic Code Understanding

The tool uses advanced semantic understanding of code:

- Code structure analysis via AST parsing
- Semantic chunking based on function/class boundaries
- Importance calculation for code chunks
- Context-aware retrieval using both semantic and keyword search

### Response Formatting

For longer responses, the tool automatically splits the output into chunks to avoid console truncation and provides proper headers and footers.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Together AI](https://www.together.ai/) for their powerful LLM API
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Rich](https://github.com/Textualize/rich) for beautiful terminal interfaces
- [Click](https://click.palletsprojects.com/) for CLI framework
