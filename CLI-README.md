# AI Code Reviewer

A powerful AI-powered code review assistant that provides real-time feedback on your codebase. This tool uses Together AI to detect critical issues, performance problems, and security vulnerabilities in your code.

## Table of Contents
1. [Installation & Setup](#installation--setup)
2. [Getting Started](#getting-started)
3. [Key Features](#key-features)
4. [Usage Guide](#usage-guide)
5. [Command Reference](#command-reference)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager
- Together AI API key

### Setup Steps

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your Together AI API key:

```bash
python cli.py configure
```

This will prompt you for your API key, which will be stored in:
- `.env` file
- `~/.codeai/config.yml`

## Getting Started

Follow these steps in order to start using the AI Code Reviewer:

### 1. Create a Project

```bash
python cli.py create-project PROJECT_NAME
```

### 2. Initialize Your Codebase

```bash
# Initialize a local codebase
python cli.py init /path/to/code

# Initialize from GitHub
python cli.py init /path/to/code --github
```

### 3. Set Project Path

```bash
python cli.py assist --set-project-path /path/to/your/project
```

### 4. Start Code Review

```bash
# Review using saved project path
python cli.py assist

# Continuously monitor for changes (watch mode)
python cli.py assist --watch
```

### 5. Ask Questions About Your Code

```bash
python cli.py ask -i "How does the authentication system work?"
```

## Key Features

### Code Review & Analysis
- 🔍 **Live Code Review**: Real-time monitoring and analysis of code changes
- 🚨 **Critical Issue Detection**: Identifies bugs, performance bottlenecks, and security risks
- 🔬 **Static Analysis**: Built-in analysis for common programming issues
- ⚠️ **Error Detection**: Comprehensive detection of:
  - Syntax errors
  - Undefined variables
  - Security vulnerabilities
  - Logic flaws
  - Potential infinite loops
  - Bare except clauses
  - Hardcoded credentials
- 📚 **Documentation Analysis**: Extract and analyze:
  - Function and class docstrings
  - Code comments
  - Code structure documentation

### Project Management
- 📂 **Multi-Project Support**: Organize and manage multiple codebases
- 🎯 **Project Path Management**: 
  - Set default project paths
  - Save project configurations
  - Switch between projects
- 🔄 **GitHub Integration**: 
  - Clone and analyze GitHub repositories
  - Support for remote codebases
- ⚙️ **Configuration Management**:
  - YAML-based settings
  - Environment variable support
  - API key management

### Processing & Performance
- ⚡ **Parallel Processing**: 
  - Multi-threaded code analysis
  - Distributed processing support
  - Batch processing capabilities
- 👀 **Watch Mode**: 
  - Continuous file monitoring
  - Real-time feedback
  - Automatic reanalysis on changes

### AI Integration
- 🤖 **Together AI**: 
  - State-of-the-art language models
  - Smart code analysis
  - Contextual understanding
- 🧠 **Smart Embeddings**: 
  - Code embedding generation
  - Semantic search
  - Context-aware analysis
- 💡 **Interactive Features**: 
  - Natural language queries
  - Code explanations
  - Suggestion generation

### Developer Tools
- 🎨 **Rich CLI Interface**: 
  - Progress bars
  - Status updates
  - Color-coded output
- 🐛 **Debugging Support**: 
  - Error tracing
  - Debug mode
  - Detailed error reporting
- 📈 **Progress Tracking**: 
  - Visual progress indicators
  - Time remaining estimates
  - Process status updates

## Usage Guide

### Understanding Review Output

The tool provides focused feedback in the following format:

```
file.py:42 - Error: Description of the critical issue
file.py:73 - Warning: Description of the potential problem
```

Issues are categorized into:
1. Critical bugs and errors
2. Major performance issues
3. Security vulnerabilities
4. Significant design flaws

### Project Management

Once you've set up your initial project, you can manage multiple projects:

#### List available projects

```bash
python cli.py list-projects
```

#### Switch to a different project

```bash
python cli.py switch-project PROJECT_NAME
```

### Additional Analysis Features

```bash
# Explain specific code sections
python cli.py explain file.py --start-line 10 --end-line 20

# Get code suggestions for specific lines
python cli.py suggest file.py --line 42

# Get code completions
python cli.py complete file.py --line 42
```

## Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `init` | **Initialize codebase (first step)** | `python cli.py init /path/to/code --github` |
| `create-project` | Create project | `python cli.py create-project myapp` |
| `configure` | Set up API key | `python cli.py configure` |
| `assist` | Start code review | `python cli.py assist --watch` |
| `ask` | Query the codebase | `python cli.py ask -i "How does X work?"` |
| `list-projects` | List all projects | `python cli.py list-projects` |
| `switch-project` | Change project | `python cli.py switch-project myapp` |
| `status` | Show statistics | `python cli.py status --format json` |
| `refresh` | Update codebase index | `python cli.py refresh --summary` |
| `complete` | Get code completions | `python cli.py complete file.py --line 42` |
| `suggest` | Get inline suggestions | `python cli.py suggest file.py --line 42` |
| `explain` | Explain code sections | `python cli.py explain file.py --start-line 10 --end-line 20` |
| `reset-history` | Clear conversation | `python cli.py reset-history` |
| `debug-projects` | Debug project registry | `python cli.py debug-projects` |
| `compose` | Manage components | `python cli.py compose add testing` |

### Command Options

#### `init` (First step)
- `--github`: Initialize from GitHub URL
- `--project`: Specify project name
- `--summary`: Generate detailed summary
- `--workers`: Number of parallel workers
- `--distributed`: Enable distributed processing

#### `assist`
- `--watch, -w`: Enable continuous monitoring
- `--set-project-path`: Set default project path

#### `ask`
- `--interactive, -i`: Start interactive mode
- `--composer, -c`: Show changes as diffs
- `--chunks`: Number of context chunks
- `--reset, -r`: Reset conversation history
- `--project, -p`: Specify project

#### `complete`
- `--line, -l`: Line number for context
- `--scope, -s`: Show scope information
- `--patterns, -p`: Show similar patterns

#### `suggest`
- `--line, -l`: Line number for suggestions
- `--scope, -s`: Show scope information
- `--threshold, -t`: Minimum relevance score

#### `explain`
- `--start-line, -s`: Start line number
- `--end-line, -e`: End line number
- `--detail, -d`: Level of detail (low/medium/high)

## Examples

### Complete Workflow Example

```bash
# 1. Create a new project
python cli.py create-project frontend-app

# 2. Initialize the codebase
python cli.py init /path/to/frontend

# 3. Set the project path
python cli.py assist --set-project-path /path/to/frontend

# 4. Start watching for changes
python cli.py assist --watch

# 5. Ask questions about the code
python cli.py ask -i "How does the router work?"
```

### Working with Multiple Projects

```bash
# Switch between projects
python cli.py switch-project backend-api

# Review current project
python cli.py assist
```

## Troubleshooting

### API Key Issues

If you encounter authentication errors:
- Run `python cli.py configure` to re-enter your API key
- Check your `.env` file and `~/.codeai/config.yml`

### Review Issues

If the review process isn't working:
- Ensure the project path is correctly set
- Check file permissions
- Verify the files are in supported languages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE) 