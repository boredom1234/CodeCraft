# AI Code Reviewer

A powerful AI-powered code review assistant that provides real-time feedback on your codebase. This tool uses Together AI to detect critical issues, performance problems, and security vulnerabilities in your code.

## Features

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

### Code Analysis
- 🏗️ **Structure Analysis**: 
  - Class and function extraction
  - Import statement analysis
  - Code pattern detection
- 📊 **Similarity Scoring**: 
  - Relevance scoring for suggestions
  - Search result ranking
  - Code chunk similarity analysis
- 🌐 **Multi-Language Support**: 
  - Python
  - JavaScript/TypeScript
  - Go
  - Additional language support

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

## Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Together AI API key

### Setup

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your Together AI API key (see Configuration section)

## Configuration

Configure your Together AI API key by running:

```bash
python cli.py configure
```

This will prompt you for your API key, which will be stored in:
- `.env` file
- `~/.codeai/config.yml`

## Usage

### Project Management

#### Create a new project

```bash
python cli.py create-project PROJECT_NAME
```

#### List available projects

```bash
python cli.py list-projects
```

#### Switch to a different project

```bash
python cli.py switch-project PROJECT_NAME
```

### Code Review Features

#### Set up project path

```bash
# Set default project path
python cli.py assist --set-project-path /path/to/your/project
```

#### Start code review

```bash
# Review using saved project path
python cli.py assist

# Review specific directory
python cli.py assist /path/to/review
```

#### Watch mode

```bash
# Continuously monitor for changes
python cli.py assist --watch
```

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

## Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `assist` | Start code review | `python cli.py assist --watch` |
| `ask` | Query the codebase | `python cli.py ask -i "How does X work?"` |
| `configure` | Set up API key | `python cli.py configure` |
| `create-project` | Create project | `python cli.py create-project myapp` |
| `init` | Initialize codebase | `python cli.py init /path/to/code --github` |
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

#### `assist`
- `--watch, -w`: Enable continuous monitoring
- `--set-project-path`: Set default project path

#### `ask`
- `--interactive, -i`: Start interactive mode
- `--composer, -c`: Show changes as diffs
- `--chunks`: Number of context chunks
- `--reset, -r`: Reset conversation history
- `--project, -p`: Specify project

#### `init`
- `--github`: Initialize from GitHub URL
- `--project`: Specify project name
- `--summary`: Generate detailed summary
- `--workers`: Number of parallel workers
- `--distributed`: Enable distributed processing

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

### Setting up a project with continuous review

```bash
# Create a new project
python cli.py create-project frontend-app

# Set the project path
python cli.py assist --set-project-path /path/to/frontend

# Start watching for changes
python cli.py assist --watch
```

### Reviewing multiple projects

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