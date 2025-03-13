# AI Codebase Reviewer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful AI-powered tool for understanding, analyzing, and exploring codebases. This tool leverages vector embeddings and Large Language Models (via Together AI) to help developers navigate and understand code through natural language interactions.

## 🌟 Key Features

- 🔍 **Intelligent Code Analysis**: Deep semantic understanding of code structure and relationships
- 💬 **Natural Language Queries**: Ask questions about your codebase in plain English
- 🎯 **Multi-Language Support**: Analyzes Python, JavaScript, TypeScript, Go, and more
- 📊 **Project Management**: Handle multiple codebases through an organized project system
- 🧠 **Context-Aware**: Maintains conversation history for coherent follow-up questions
- 🔄 **GitHub Integration**: Directly analyze repositories from GitHub
- 🎨 **Rich CLI Interface**: Beautiful, interactive command-line interface with progress tracking

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Together AI API key ([Get one here](https://www.together.ai/))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/boredom1234/ai-codebase-reviewer.git
cd ai-codebase-reviewer
```

2. Create and activate a virtual environment (recommended):
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

### Initialize a Project

```bash
# Create a new project
python cli.py create-project my-project

# Index a local codebase
python cli.py init /path/to/codebase --project my-project

# Or index from GitHub
python cli.py init https://github.com/user/repo --github --project my-project
```

### Ask Questions

```bash
# Interactive mode with conversation history
python cli.py ask --interactive --project my-project

# Single question mode
python cli.py ask --project my-project

# With code changes suggestions
python cli.py ask --composer --project my-project
```

### Manage Projects

```bash
# List all projects
python cli.py list-projects

# Switch between projects
python cli.py switch-project another-project

# View project status
python cli.py status
```

## 🛠️ Configuration

The tool can be configured through:

1. Environment variables (`.env` file):
```env
TOGETHER_API_KEY=your_api_key_here
```

2. Project configuration (`~/.codeai/config.yml`):
```yaml
chunk_size: 20
overlap: 5
max_history: 5
temperature: 0.7
debug: false
```

## 📚 Command Reference

| Command | Description | Options |
|---------|-------------|---------|
| `configure` | Set up API keys and settings | - |
| `create-project` | Create/switch to a project | `PROJECT_NAME` |
| `init` | Initialize codebase | `--github`, `--project` |
| `ask` | Query the codebase | `--interactive`, `--composer`, `--chunks`, `--reset`, `--project` |
| `list-projects` | Show all projects | - |
| `switch-project` | Change active project | `PROJECT_NAME` |
| `status` | Show project statistics | `--format [text\|json]` |
| `reset-history` | Clear conversation history | - |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:
- Submit bug reports and feature requests
- Set up your development environment
- Submit pull requests
- Follow our coding standards

## 🔍 How It Works

The tool uses several advanced techniques:
1. **Code Structure Analysis**: Parses and understands code structure using AST and regex
2. **Vector Embeddings**: Creates semantic embeddings of code chunks for intelligent search
3. **LLM Integration**: Uses Together AI's LLMs for natural language understanding
4. **Context Management**: Maintains conversation history and code context for better responses

## 🐛 Troubleshooting

### Common Issues

1. **API Key Problems**:
   - Ensure your Together AI API key is correctly configured
   - Run `python cli.py configure` to reconfigure

2. **Indexing Issues**:
   - Check file permissions
   - Ensure sufficient disk space
   - Verify supported file types

3. **Query Problems**:
   - Be more specific in questions
   - Use `--chunks` to adjust context size
   - Try `reset-history` if conversation derails

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Together AI](https://www.together.ai/) for their powerful LLM API
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Rich](https://github.com/Textualize/rich) for beautiful terminal interfaces
- [Click](https://click.palletsprojects.com/) for CLI framework 
