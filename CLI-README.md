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

Projects help you organize different codebases and maintain separate conversation histories.

### 2. Initialize Your Codebase

```bash
# Initialize a local codebase
python cli.py init /path/to/code

# Initialize from GitHub
python cli.py init https://github.com/user/repo --github
```

This step indexes your codebase for semantic search and analysis.

### 3. Start Using the Tool

You can now interact with your codebase in various ways:

```bash
# Ask questions about your code
python cli.py ask -i "How does the authentication system work?"

# Get live code reviews
python cli.py assist --watch

# Get code suggestions
python cli.py suggest file.py --line 42
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
- 🧠 **Memory-Aware Processing**:
  - Adaptive batch sizing based on system memory
  - Automatic resource optimization
  - Stream results to disk to avoid memory issues
- 🌐 **Distributed Processing**:
  - Support for remote worker nodes
  - Scalable analysis for large codebases
  - Worker management system

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
- 📋 **Response Formatting**:
  - Structured outputs with headers and footers
  - Chunked responses for large outputs
  - Rich markdown formatting
- 🔄 **Model Selection**:
  - Switch between LLM models
  - Configure default models
  - Model-specific parameters

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

### Understanding Commands and Workflow

The tool follows a typical workflow:

1. **Configure**: Set up your API key
2. **Create Project**: Organize your codebase
3. **Initialize**: Index your code for analysis
4. **Interact**: Use various commands to analyze, query, and improve your code

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

Here is a comprehensive list of all available commands:

### Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `configure` | Set up API key and settings | `python cli.py configure` |
| `create-project` | Create a new project | `python cli.py create-project myapp` |
| `init` | Initialize and index a codebase | `python cli.py init /path/to/code` |
| `ask` | Ask questions about the codebase | `python cli.py ask -i "How does X work?"` |

### Project Management Commands

| Command | Description | Example |
|---------|-------------|---------|
| `list-projects` | List all available projects | `python cli.py list-projects` |
| `switch-project` | Switch to an existing project | `python cli.py switch-project myapp` |
| `debug-projects` | Debug project registry | `python cli.py debug-projects` |
| `status` | Show current project status and statistics | `python cli.py status --format json` |

### Code Analysis Commands

| Command | Description | Example |
|---------|-------------|---------|
| `assist` | Live code review assistant | `python cli.py assist --watch` |
| `complete` | Get AI-powered code completions | `python cli.py complete file.py --line 42` |
| `suggest` | Get intelligent inline code suggestions | `python cli.py suggest file.py --line 42` |
| `explain` | Get natural language explanation of code | `python cli.py explain file.py --start-line 10 --end-line 20` |

### Maintenance Commands

| Command | Description | Example |
|---------|-------------|---------|
| `refresh` | Refresh the codebase index | `python cli.py refresh --project myproject` |
| `reset-history` | Reset conversation history | `python cli.py reset-history` |
| `compose` | Manage project components and dependencies | `python cli.py compose add testing` |
| `set-model` | Set default LLM model | `python cli.py set-model --list` |

### Detailed Command Options

#### `configure`
Configures API keys and settings.

```bash
python cli.py configure
```

#### `create-project <project_name>`
Creates a new project or switches to an existing one.

```bash
python cli.py create-project my_new_project
```

#### `init <source>`
Initializes and indexes a codebase.

Options:
- `--github`: Treat source as GitHub URL
- `--project, -p <project>`: Project name to store index in
- `--summary`: Generate detailed codebase summary
- `--example`: Show example usage
- `--workers <workers>`: Number of parallel workers
- `--batch-size <batch_size>`: Batch size for processing files (default: 10)
- `--distributed`: Enable distributed processing
- `--worker-addresses <worker_addresses>`: Comma-separated list of worker addresses for distributed processing

Examples:
```bash
# Initialize a local directory with parallel processing
python cli.py init /path/to/codebase --workers 4 --batch-size 20

# Initialize from a GitHub repository with distributed processing
python cli.py init https://github.com/user/repo --github --distributed --worker-addresses host1:5000,host2:5000
```

#### `ask`
Ask questions about the codebase.

Options:
- `--interactive, -i`: Start interactive mode
- `--composer, -c`: Show suggested changes as diffs
- `--chunks <chunks>`: Number of code chunks to use for context
- `--reset, -r`: Reset conversation history before starting
- `--project, -p <project>`: Project to use
- `--concise`: Force concise responses (1-2 sentences)
- `--model, -m <model>`: Specific model to use for this session

Examples:
```bash
# Interactive mode
python cli.py ask --interactive

# Show suggested changes as diffs
python cli.py ask --composer

# Set context size and reset history
python cli.py ask --chunks 5 --reset

# Use specific model for a session
python cli.py ask -i --model meta-llama/Llama-3.3-70B-Instruct-Turbo
```

#### `list-projects`
Lists all available projects.

```bash
python cli.py list-projects
```

#### `switch-project <project_name>`
Switches to an existing project.

```bash
python cli.py switch-project backend_api
```

#### `status`
Shows current project status and statistics.

Options:
- `--format, -f <format>`: Output format (text|json) for the status information

Examples:
```bash
# Default text format
python cli.py status

# JSON format
python cli.py status --format json
```

#### `refresh`
Refreshes the codebase index by updating only changed files.

Options:
- `--project, -p <project>`: Project to refresh
- `--summary`: Generate detailed codebase summary after refresh

Examples:
```bash
# Refresh current project
python cli.py refresh

# Refresh specific project with summary
python cli.py refresh --project myproject --summary
```

#### `reset-history`
Resets conversation history while keeping codebase index.

```bash
python cli.py reset-history
```

#### `assist`
Live code review assistant.

Options:
- `--watch, -w`: Watch mode - continuously review code changes
- `--set-project-path <set_project_path>`: Set and save the project path for future use

Examples:
```bash
# Set project path for future use
python cli.py assist --set-project-path /path/to/your/project

# Review using saved project path
python cli.py assist

# Watch mode for continuous review
python cli.py assist --watch
```

#### `complete <file_path>`
Get AI-powered code completions with full codebase context.

Options:
- `--line, -l <line>`: Line number for context
- `--scope, -s`: Show current scope information
- `--patterns, -p`: Show similar code patterns

Examples:
```bash
# Get completion at specific line
python cli.py complete path/to/file.py --line 42

# Show scope and similar patterns
python cli.py complete path/to/file.py --line 42 --scope --patterns
```

#### `suggest <file_path>`
Get intelligent inline code suggestions with codebase awareness.

Options:
- `--line, -l <line>`: Line number to get suggestions for (required)
- `--scope, -s`: Show current scope information
- `--threshold, -t <threshold>`: Minimum relevance score (0-1)

Examples:
```bash
# Get suggestions with default threshold
python cli.py suggest path/to/file.py --line 42

# Show scope and require high relevance
python cli.py suggest path/to/file.py --line 42 --scope --threshold 0.8
```

#### `explain <file_path>`
Get natural language explanation of code.

Options:
- `--start-line, -s <start_line>`: Start line number
- `--end-line, -e <end_line>`: End line number
- `--detail, -d <detail>`: Level of detail in explanation (low|medium|high)

Examples:
```bash
# Explain entire file with medium detail
python cli.py explain path/to/file.py

# Explain specific lines with high detail
python cli.py explain path/to/file.py --start-line 10 --end-line 20 --detail high
```

#### `debug-projects`
Debug project registry (for development use).

```bash
python cli.py debug-projects
```

#### `compose <action> [component]`
Manage project components and dependencies.

Arguments:
- `<action>`: Action to perform (add|remove|list)
- `[component]`: Component to act on (required for add/remove)

Examples:
```bash
# List available components
python cli.py compose list

# Add testing component
python cli.py compose add testing

# Remove logging component
python cli.py compose remove logging
```

#### `set-model`
Set default LLM model.

Options:
- `--list`: List available models
- `--model <model>`: Specific model to set as default

Examples:
```bash
# List available models
python cli.py set-model --list

# Set a different default model
python cli.py set-model meta-llama/Llama-3-70B-Instruct
```

## Examples

### Complete Workflow Examples

#### Basic Workflow

```bash
# 1. Configure API key
python cli.py configure

# 2. Create a new project
python cli.py create-project frontend-app

# 3. Initialize the codebase
python cli.py init /path/to/frontend --project frontend-app

# 4. Ask questions about the code
python cli.py ask -i "How does the router work?"
```

#### Advanced Analysis Workflow

```bash
# 1. Set up project
python cli.py create-project backend-api
python cli.py init /path/to/backend --project backend-api

# 2. Check project status
python cli.py status

# 3. Set project path for continuous review
python cli.py assist --set-project-path /path/to/backend

# 4. Start continuous review
python cli.py assist --watch

# 5. In another terminal, get specific suggestions
python cli.py suggest /path/to/backend/auth/login.py --line 42 --scope
```

#### Working with Multiple Projects

```bash
# List all projects
python cli.py list-projects

# Switch between projects
python cli.py switch-project backend-api

# Refresh index after code changes
python cli.py refresh

# Check status in JSON format
python cli.py status --format json
```

#### Code Enhancement Workflow

```bash
# Get detailed explanation of a complex function
python cli.py explain /path/to/complex_module.py --start-line 120 --end-line 150 --detail high

# Get completion suggestions
python cli.py complete /path/to/module.py --line 75 --patterns

# Get suggestions with high relevance
python cli.py suggest /path/to/module.py --line 30 --threshold 0.8
```

#### Component Management

```bash
# List available components
python cli.py compose list

# Add testing infrastructure
python cli.py compose add testing

# Start interactive session with code composer
python cli.py ask --interactive --composer
```

## Troubleshooting

### API Key Issues

If you encounter authentication errors:
- Run `python cli.py configure` to re-enter your API key
- Check your `.env` file and `~/.codeai/config.yml`

### Project Registry Problems

If you're having issues with projects:
- Use `python cli.py debug-projects` to inspect the registry
- Try recreating the project with `python cli.py create-project <n>`

### Indexing Issues

If initialization is failing:
- Check file permissions
- Ensure your codebase is accessible
- Try running with fewer workers: `python cli.py init /path/to/code --workers 1`
- Check system memory if receiving out-of-memory errors

### Watch Mode Problems

If watch mode isn't detecting changes:
- Verify the project path is set correctly
- Check file permissions
- Make sure files have supported extensions

### Memory Usage Issues

If encountering memory problems:
- Enable disk streaming in config.yml:
  ```yaml
  parallel_processing:
    memory_limit_percentage: 70.0
    stream_to_disk: true
    cache_dir: .cache
  ```
- Reduce the batch size for processing: `--batch-size 5`
- Use distributed processing for very large codebases

### Model Selection

If having issues with specific models:
- List available models: `python cli.py set-model --list`
- Set a different default model: `python cli.py set-model meta-llama/Llama-3-70B-Instruct`
- Check for model availability in the Together AI platform

## Advanced Configuration

You can customize the tool's behavior by modifying the `config.yml` file:

```yaml
# Model settings
model:
  default: meta-llama/Llama-3.3-70B-Instruct-Turbo
  max_tokens: 4000
  concise_responses: false

# Parallel processing settings
parallel_processing:
  max_workers: 4
  memory_limit_percentage: 80.0
  stream_to_disk: true
  cache_dir: .cache

# Distributed processing
distributed:
  enabled: false
  workers:
    - worker1:5000
    - worker2:5000
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE) 