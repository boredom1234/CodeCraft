# AI Code Reviewer

A powerful AI-powered command-line tool for understanding, analyzing, and exploring codebases. This tool uses vector embeddings and LLMs (via Together AI) to help developers navigate and understand code.

## Features

- 🔍 **Code Indexing**: Index local or GitHub repositories for AI-powered search
- 💬 **Intelligent Querying**: Ask questions about your codebase in natural language
- 📊 **Multiple Projects**: Manage different codebases through the project system
- 🧠 **Contextual Memory**: Maintains conversation history for better follow-up questions
- 🔄 **GitHub Integration**: Pull and analyze repositories directly from GitHub
- 📈 **Similarity Scores**: Shows relevance percentages for files and code chunks to prioritize important information

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your Together AI API key (see Configuration section)

## Configuration

Before using the tool, you need to configure your Together AI API key. You can do this by running:

```bash
python cli.py configure
```

This will prompt you for your Together AI API key, which will be stored securely in a `.env` file and in `~/.codeai/config.yml`.

Alternatively, you can manually create a `.env` file with:

```
TOGETHER_API_KEY=your_api_key_here
```

## Usage

### Project Management

The tool uses a project system to manage different codebases.

#### Create a new project

```bash
python cli.py create-project PROJECT_NAME
```

This creates a new project and sets it as the active project.

#### List available projects

```bash
python cli.py list-projects
```

Shows all projects with their status (active/inactive) and creation date.

#### Switch to a different project

```bash
python cli.py switch-project PROJECT_NAME
```

Changes the active project to the specified one.

### Indexing a Codebase

#### Index a local directory

```bash
python cli.py init PATH_TO_DIRECTORY [--project PROJECT_NAME]
```

This scans and indexes the specified directory, making it ready for querying.

#### Index a GitHub repository

```bash
python cli.py init GITHUB_REPO_URL --github [--project PROJECT_NAME]
```

This clones and indexes the specified GitHub repository.

#### Refresh a codebase index

```bash
python cli.py refresh [--project PROJECT_NAME] [--summary]
```

This detects new, modified, and deleted files in the codebase and updates the index accordingly. It's much faster than re-indexing everything, especially for large codebases after small changes.

### Querying the Codebase

#### Ask a single question

```bash
python cli.py ask [--project PROJECT_NAME]
```

You'll be prompted to enter your question about the codebase.

#### Interactive mode

```bash
python cli.py ask --interactive [--project PROJECT_NAME]
```

Start an interactive session where you can ask multiple questions with persistent conversation history.

#### Control context size

```bash
python cli.py ask --chunks NUMBER_OF_CHUNKS
```

Control how many chunks of code are used for context (default is dynamic based on question).

#### Reset conversation history

```bash
python cli.py reset-history
```

Clears the conversation history while keeping the codebase index.

### Understanding Similarity Scores

When you ask a question, the tool now displays relevance percentages for files and code chunks:

```
Searching for relevant code to answer: How does the authentication system work?
Using 8 chunks for context
Found 8 relevant code sections
File: auth/login.py Relevance: 82.17%
  Chunk lines 10-35 Relevance: 89.54%
  Chunk lines 40-60 Relevance: 74.80%
File: models/user.py Relevance: 65.32%
  Chunk lines 15-30 Relevance: 68.75%
```

This helps you:
1. Understand which files and code sections are most relevant to your question
2. See how the tool prioritizes information (sorted by relevance)
3. Focus on the highest-relevance code first

The similarity scores are calculated based on both semantic relevance and code structure.

## Examples

### Creating and setting up a project

```bash
# Create a new project
python cli.py create-project frontend-app

# Index a GitHub repository
python cli.py init https://github.com/user/frontend-app --github

# Ask questions about the codebase
python cli.py ask --interactive
```

### Switching between projects

```bash
# List available projects
python cli.py list-projects

# Switch to another project
python cli.py switch-project backend-api

# Ask questions about the newly activated project
python cli.py ask
```

## Command Reference

| Command | Description |
|---------|-------------|
| `configure` | Set up API keys and configuration |
| `create-project PROJECT_NAME` | Create a new project or switch to existing one |
| `init SOURCE [--github] [--project PROJECT_NAME]` | Initialize and index a codebase |
| `list-projects` | List all available projects |
| `switch-project PROJECT_NAME` | Switch to an existing project |
| `ask [--interactive] [--chunks NUM] [--reset] [--project PROJECT_NAME]` | Ask questions about the codebase |
| `reset-history` | Reset conversation history while keeping the index |

## Troubleshooting

### API Key Issues

If you encounter authentication errors:
- Ensure your Together AI API key is correctly set
- Try running `python cli.py configure` to re-enter your API key

### Indexing Issues

If indexing fails:
- Check if the repository or directory exists and is accessible
- Ensure you have sufficient disk space
- Check the file types in your codebase (the tool indexes common code file types)

### Query Issues

If you get poor query results:
- Try to be more specific in your questions
- Use the `--chunks` option to include more context
- Reset the conversation history with `reset-history` if the conversation has gone off track

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE) 