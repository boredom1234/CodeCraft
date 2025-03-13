# cli.py
import click
import os
import asyncio
import datetime
import pickle
import traceback
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
import yaml
from dotenv import load_dotenv
from analyzer import CodebaseAnalyzer

console = Console()

def validate_config(config: dict) -> dict:
    """Validate and set defaults for configuration"""
    defaults = {
        "chunk_size": 20,
        "overlap": 5,
        "max_history": 5,
        "temperature": 0.7,
        "debug": False
    }
    
    if not isinstance(config, dict):
        config = {}
    
    # Merge with defaults
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            
    return config

def setup_together_api():
    """Setup Together AI API key"""
    # First try to load from .env
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")
    if api_key:
        return api_key
    
    # Then try to load from config.yml
    config_dir = Path.home() / ".codeai"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.yml"
    
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
            if config.get("together_api_key"):
                return config["together_api_key"]
    
    # If no key found, prompt user
    api_key = Prompt.ask(
        "Please enter your Together AI API key",
        password=True
    )
    
    # Save to both .env and config.yml
    env_file = Path(".env")
    with open(env_file, "a") as f:
        f.write(f"\nTOGETHER_API_KEY={api_key}")
    
    # Add config validation
    config = {"together_api_key": api_key}
    config = validate_config(config)
    
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    
    return api_key

@click.group()
def cli():
    """Code Understanding Assistant powered by Together AI

    This tool helps you analyze codebases, providing insights, suggestions,
    and debugging assistance. Use the commands below to interact with the tool.
    """

# Add a more robust error handling decorator
def handle_cli_errors(f):
    """Decorator to handle CLI command errors consistently"""
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/]")
            exit(0)
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/]")
            if os.getenv("DEBUG"):
                traceback.print_exc()
            exit(1)
    return wrapper

# Apply to CLI commands
@cli.command()
@handle_cli_errors
def configure():
    """Configure API keys and settings

    This command helps you set up the necessary API keys and configuration
    settings for the application. It will prompt you to enter your Together AI
    API key if it's not already set.
    """
    api_key = setup_together_api()
    console.print("[green]Configuration saved successfully![/]")

def load_config() -> dict:
    """Load configuration from config.yml file."""
    config_path = Path('config.yml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

# Load configuration at the start
config = load_config()

# Use configuration settings in the application
# For example, setting default chunk size from config
chunk_size = config.get('chunk_size', 20)  # Default to 20 if not set

@cli.command()
@click.argument('source')
@click.option('--github', is_flag=True, help='Treat source as GitHub URL')
@click.option('--project', '-p', help='Project name to store index in')
@click.option('--summary', is_flag=True, help='Generate detailed codebase summary (may be slow for large codebases)')
@click.option('--example', is_flag=True, help='Show example usage')
def init(source: str, github: bool, project: str = None, summary: bool = False, example: bool = False):
    """Initialize and index a codebase.

    This command initializes a codebase for analysis. You can specify a local
    directory or a GitHub repository URL. Use the --github flag if the source
    is a GitHub URL. Optionally, specify a project name to store the index.

    Examples:
    
    
    # Initialize a local directory
    python cli.py init /path/to/codebase

    # Initialize from a GitHub repository
    python cli.py init https://github.com/user/repo --github
    """
    try:
        # Show example usage if requested
        if example:
            console.print("[bold blue]Example Usage:[/]")
            console.print("\n[green]Initialize local directory:[/]")
            console.print("  python cli.py init /path/to/codebase")
            console.print("\n[green]Initialize from GitHub:[/]")
            console.print("  python cli.py init https://github.com/user/repo --github")
            return

        # Ensure API key is configured
        api_key = setup_together_api()
        console.print(f"[green]Using API key:[/] {api_key[:5]}...{api_key[-5:]}")
        
        # Set up project if specified
        if project:
            create_project_cmd = get_command('create_project')
            ctx = click.Context(create_project_cmd, info_name='create_project')
            create_project(ctx, project_name=project)
        
        async def run_setup_and_indexing():
            try:
                # Initialize the analyzer with Together AI client
                if github:
                    console.print(f"[bold]Initializing from GitHub repository:[/] {source}")
                    analyzer = await CodebaseAnalyzer.from_github(source, api_key, project)
                else:
                    console.print(f"[bold]Initializing from local directory:[/] {source}")
                    analyzer = CodebaseAnalyzer(source, api_key, project)
                
                # Configure the analyzer
                analyzer.config['chunk_size'] = chunk_size
                analyzer.config['generate_summary'] = summary
                
                if summary:
                    console.print("[bold yellow]Detailed codebase summary generation enabled (this may take longer)[/]")
                
                # Index the codebase
                await analyzer.index()
                
                # Save analyzer state
                console.print("[green]Saving analyzer state...[/]")
                save_analyzer_state(analyzer)
                console.print(f"[bold green]✓ Codebase indexed successfully in project '{analyzer.project_name}'![/]")
                
            except Exception as e:
                console.print(f"[bold red]Error in async operation: {str(e)}[/]")
                traceback.print_exc()
                raise
        
        # Run the async function
        console.print("[bold blue]Starting async operation...[/]")
        asyncio.run(run_setup_and_indexing())
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()
        exit(1)

@cli.command()
@click.option('--interactive', '-i', is_flag=True, help='Start interactive mode')
@click.option('--composer', '-c', is_flag=True, help='Show suggested changes as diffs')
@click.option('--chunks', type=int, help='Number of code chunks to use for context')
@click.option('--reset', '-r', is_flag=True, help='Reset conversation history before starting')
@click.option('--project', '-p', help='Project to use')
def ask(interactive: bool, composer: bool = False, chunks: int = None, reset: bool = False, project: str = None):
    """Ask questions about the codebase.

    This command allows you to query the codebase for information or
    suggestions. You can use it in interactive mode or provide a single
    question.

    Examples:
        codeai ask --interactive
        codeai ask "What does this function do?"
    """
    try:
        # Change to specified project if requested
        if project:
            create_project_cmd = get_command('create_project')
            ctx = click.Context(create_project_cmd, info_name='create_project')
            create_project_cmd.invoke(ctx, project_name=project)
        
        console.print("[bold blue]Loading code analyzer...[/]")
        analyzer = load_analyzer_state(project)
        
        if reset:
            # Reset conversation history if requested
            analyzer.conversation_history = []
            save_analyzer_state(analyzer)
            console.print("[bold green]✓ Conversation history reset successfully![/]")
        elif analyzer.conversation_history:
            console.print(f"[green]Loaded {len(analyzer.conversation_history)} previous conversation exchanges[/]")
        
        async def handle_question(question: str):
            try:
                with console.status("[bold yellow]Thinking...[/]"):
                    if composer:
                        # Get response in composer mode
                        response = await analyzer.query(
                            f"Act as a code composer. For the following request, show the exact changes needed using + for additions and - for removals: {question}",
                            chunks
                        )
                        
                        # Process response to highlight diffs
                        lines = response.split('\n')
                        formatted_lines = []
                        in_code_block = False
                        current_file = None
                        changes = {}  # Store changes by file
                        
                        for line in lines:
                            # Check for file markers in code blocks
                            if line.strip().startswith('```python:'):
                                in_code_block = True
                                current_file = line.split(':', 1)[1].strip()
                                changes[current_file] = {'additions': [], 'removals': []}
                                formatted_lines.append(line)
                            elif line.strip() == '```':
                                in_code_block = False
                                current_file = None
                                formatted_lines.append(line)
                            elif in_code_block and current_file:
                                # Store changes for each file
                                if line.startswith('+'):
                                    changes[current_file]['additions'].append(line[1:].strip())
                                    formatted_lines.append(f"[green]{line}[/]")
                                elif line.startswith('-'):
                                    changes[current_file]['removals'].append(line[1:].strip())
                                    formatted_lines.append(f"[red]{line}[/]")
                                else:
                                    formatted_lines.append(line)
                            else:
                                formatted_lines.append(line)
                        
                        # Show the formatted response
                        console.print(f"\n[bold blue]Proposed Changes:[/]")
                        console.print('\n'.join(formatted_lines))
                        
                        # If we have changes, ask for confirmation
                        if changes:
                            console.print("\n[bold blue]Files to be modified:[/]")
                            for file_path in changes.keys():
                                console.print(f"  • {file_path}")
                            
                            if Confirm.ask("\n[bold yellow]Do you want to apply these changes?"):
                                # Apply the changes
                                for file_path, file_changes in changes.items():
                                    try:
                                        file_path = Path(file_path)
                                        
                                        # Create parent directories if they don't exist
                                        file_path.parent.mkdir(parents=True, exist_ok=True)
                                        
                                        # Read existing content or create new file
                                        content = []
                                        if file_path.exists():
                                            with open(file_path, 'r') as f:
                                                content = f.readlines()
                                        
                                        # Remove lines that should be removed
                                        content = [line for line in content 
                                                 if line.strip() not in file_changes['removals']]
                                        
                                        # Add new lines
                                        content.extend(line + '\n' for line in file_changes['additions'])
                                        
                                        # Write back the modified content
                                        with open(file_path, 'w') as f:
                                            f.writelines(content)
                                        
                                        console.print(f"[green]✓ Updated {file_path}[/]")
                                        
                                    except Exception as e:
                                        console.print(f"[red]Error modifying {file_path}: {str(e)}[/]")
                                
                                console.print("[bold green]✓ Changes applied successfully![/]")
                            else:
                                console.print("[yellow]Changes rejected - no modifications made[/]")
                    else:
                        # Normal mode
                        response = await analyzer.query(question, chunks)
                        # The response is already formatted and printed by the _format_markdown_response method
                        # so we don't need to print it again here
                    
                    # Save analyzer state to persist conversation history
                    save_analyzer_state(analyzer)
                    
            except Exception as e:
                console.print(f"[bold red]Error generating response: {str(e)}[/]")
                traceback.print_exc()
        
        if interactive:
            console.print("[bold yellow]Interactive mode (Ctrl+C to exit)[/]")
            if composer:
                console.print("[bold yellow]Composer mode enabled - showing changes as diffs[/]")
            console.print("[dim]Conversation history is maintained between questions[/]\n")
            try:
                while True:
                    question = Prompt.ask("[bold green]Your question")
                    asyncio.run(handle_question(question))
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/]")
        else:
            question = Prompt.ask("[bold green]Your question")
            asyncio.run(handle_question(question))
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()
        exit(1)

@cli.command()
def reset_history():
    """Reset conversation history while keeping codebase index.

    This command clears the conversation history for the active project,
    allowing you to start fresh without re-indexing the codebase.
    """
    try:
        console.print("[bold blue]Loading code analyzer...[/]")
        analyzer = load_analyzer_state()
        
        # Reset conversation history
        analyzer.conversation_history = []
        save_analyzer_state(analyzer)
        console.print("[bold green]✓ Conversation history reset successfully![/]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()
        exit(1)

def save_analyzer_state(analyzer):
    """Save analyzer state to disk"""
    analyzer.save_state()

def load_analyzer_state(project_name: str = None):
    """Load analyzer state from disk"""
    # Get API key
    api_key = setup_together_api()
    
    # Create analyzer with saved state
    analyzer = CodebaseAnalyzer(".", api_key, project_name)  # Path doesn't matter for loading
    
    try:
        analyzer.load_state()
        return analyzer
    except FileNotFoundError:
        console.print("[yellow]No saved state found. Please index a codebase first using 'python cli.py init'[/]")
        exit(1)
    except Exception as e:
        console.print(f"[bold red]Error loading analyzer state: {str(e)}[/]")
        traceback.print_exc()
        exit(1)

def get_command(command_name):
    """Get a command from the CLI group by name"""
    return cli.get_command(None, command_name)

@cli.command()
@click.argument('project_name')
def create_project(project_name: str):
    """Create a new project or switch to an existing one.

    This command creates a new project or switches to an existing one.
    It manages the project registry and sets the specified project as active.

    Examples:
        codeai create-project my_new_project
        codeai switch-project existing_project
    """
    try:
        projects_dir = Path(".codeai") / "projects"
        projects_dir.mkdir(exist_ok=True, parents=True)
        
        # Create project directory
        project_dir = projects_dir / project_name
        project_dir.mkdir(exist_ok=True)
        
        # Create or update project registry
        registry_file = Path(".codeai") / "registry.pkl"
        projects = {}
        
        if registry_file.exists():
            with open(registry_file, 'rb') as f:
                projects = pickle.load(f)
        
        # Update active project
        projects['active'] = project_name
        if project_name not in projects:
            projects[project_name] = {"created": str(datetime.datetime.now())}
            
        with open(registry_file, 'wb') as f:
            pickle.dump(projects, f)
            
        console.print(f"[bold green]✓ Project '{project_name}' is now active[/]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()
        exit(1)

@cli.command()
def list_projects():
    """List all available projects.

    This command lists all the projects that have been created and
    indicates which project is currently active.

    Example:
        codeai list-projects
    """
    try:
        registry_file = Path(".codeai") / "registry.pkl"
        console.print(f"[dim]Looking for registry at: {registry_file}[/]")
        
        if not registry_file.exists():
            console.print("[yellow]No projects found. Create one with 'create-project'[/]")
            return
            
        console.print("[dim]Registry file exists, loading...[/]")
        with open(registry_file, 'rb') as f:
            projects = pickle.load(f)
        
        console.print(f"[dim]Loaded projects: {projects}[/]")
        active = projects.get('active', None)
        console.print(f"[dim]Active project: {active}[/]")
        
        # Count non-active entries (actual projects)
        project_count = sum(1 for name in projects if name != 'active')
        console.print(f"[dim]Project count: {project_count}[/]")
        
        if project_count == 0:
            console.print("[yellow]No projects found. Create one with 'create-project'[/]")
            return
            
        console.print("[bold blue]═════════════════════════════[/]")
        console.print("[bold blue]       AVAILABLE PROJECTS     [/]")
        console.print("[bold blue]═════════════════════════════[/]")
        
        # Process projects in a sorted way
        project_names = sorted([name for name in projects.keys() if name != 'active'])
        
        for i, name in enumerate(project_names):
            info = projects[name]
            console.print(f"[dim]Processing project: {name}[/]")
            
            # Determine if this is the active project
            status = "[bold green]ACTIVE  [/]" if name == active else "[dim]INACTIVE[/]"
            created = info.get('created', 'unknown')
            
            # Print project information with clear formatting
            console.print(f"[bold]Project {i+1}:[/] {name}")
            console.print(f"  Status: {status}")
            console.print(f"  Created: {created}")
            
            # Add separator between projects
            if i < len(project_names) - 1:
                console.print("[blue]─────────────────────────────[/]")
                
        console.print("[bold blue]═════════════════════════════[/]")
        console.print(f"[bold green]Total projects: {project_count}[/]")
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()

@cli.command()
def debug_projects():
    """Debug project registry (for development use).

    This command is intended for developers to debug the project registry.
    It displays raw registry content and attempts to recreate the registry
    if necessary.

    Example:
        codeai debug-projects
    """
    try:
        registry_file = Path(".codeai") / "registry.pkl"
        
        if not registry_file.exists():
            console.print("[yellow]Registry file does not exist![/]")
            return
            
        with open(registry_file, 'rb') as f:
            projects = pickle.load(f)
        
        console.print(f"[bold blue]Raw registry content:[/] {projects}")
        
        active = projects.get('active', None)
        console.print(f"[bold blue]Active project:[/] {active}")
        
        console.print("\n[bold blue]All projects:[/]")
        project_count = 0
        for name, info in projects.items():
            if name != 'active':
                project_count += 1
                console.print(f"  - Project name: '{name}'")
                console.print(f"    Is active: {name == active}")
                console.print(f"    Created: {info.get('created', 'unknown')}")
                console.print(f"    Raw info: {info}")
        
        console.print(f"\n[bold blue]Total projects found:[/] {project_count}") 
        
        # Try again with direct keys/values
        console.print("\n[bold blue]Direct keys/values from registry:[/]")
        for key in projects.keys():
            console.print(f"Key: {key}")
        
        # Recreate the registry
        console.print("\n[bold blue]Recreating registry...[/]")
        new_registry = {}
        new_registry['active'] = active
        for name, info in projects.items():
            if name != 'active':
                new_registry[name] = info
                console.print(f"Added {name} to new registry")
        
        with open(registry_file, 'wb') as f:
            pickle.dump(new_registry, f)
        console.print("[green]Registry recreated successfully[/]")
                
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()

@cli.command()
@click.argument('project_name')
def switch_project(project_name: str):
    """Switch to an existing project"""
    try:
        registry_file = Path(".codeai") / "registry.pkl"
        
        if not registry_file.exists():
            console.print("[yellow]No projects found. Create one with 'create-project' first.[/]")
            return
            
        with open(registry_file, 'rb') as f:
            projects = pickle.load(f)
        
        # Check if project exists
        if project_name not in projects or project_name == 'active':
            console.print(f"[bold red]Error: Project '{project_name}' does not exist.[/]")
            console.print("[yellow]Use 'list-projects' to see available projects or 'create-project' to create a new one.[/]")
            return
        
        # Update active project
        projects['active'] = project_name
        
        with open(registry_file, 'wb') as f:
            pickle.dump(projects, f)
            
        console.print(f"[bold green]✓ Switched to project '{project_name}'[/]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()
        exit(1)

@cli.command()
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text', help='Output format for the status information')
def status(format='text'):
    """Show current project status and statistics.

    This command displays the current status of the active project,
    including the number of indexed files, total chunks, and conversation
    history length.

    Examples:
        codeai status --format text
        codeai status --format json
    """
    try:
        analyzer = load_analyzer_state()
        
        stats = {
            "project": analyzer.project_name,
            "indexed_files": len(set(m['file'] for m in analyzer.metadata)),
            "total_chunks": len(analyzer.documents),
            "conversation_history": len(analyzer.conversation_history),
            "embedding_dimension": analyzer.faiss_index.d if analyzer.faiss_index else None
        }
        
        if format == 'json':
            console.print_json(data=stats)
        else:
            console.print("[bold blue]Project Status[/]")
            for key, value in stats.items():
                console.print(f"{key}: {value}")
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")

@cli.command()
@click.argument('action', type=click.Choice(['add', 'remove', 'list']))
@click.argument('component', required=False)
def compose(action: str, component: str = None):
    """Manage project components and dependencies.

    This command allows you to add, remove, or list project components
    and dependencies. Use 'add' to include a new component, 'remove' to
    delete an existing component, or 'list' to view available components.

    Examples:
        codeai compose add testing
        codeai compose remove logging
        codeai compose list
    """
    try:
        if action == 'list':
            _show_available_components()
            return
            
        if not component:
            console.print("[red]Error: Component name is required for add/remove actions[/]")
            return
            
        if action == 'add':
            _show_component_diff(component, adding=True)
        else:  # remove
            _show_component_diff(component, adding=False)
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()

def _show_available_components():
    """Show available components that can be added"""
    components = {
        "testing": {
            "description": "Add testing infrastructure with pytest",
            "files": ["tests/", "pytest.ini", "requirements-test.txt"],
            "dependencies": ["pytest", "pytest-asyncio", "pytest-cov"]
        },
        "logging": {
            "description": "Add structured logging support",
            "files": ["logging_config.py"],
            "dependencies": ["structlog"]
        },
        "docker": {
            "description": "Add Docker support for containerization",
            "files": ["Dockerfile", "docker-compose.yml", ".dockerignore"],
            "dependencies": []
        }
    }
    
    console.print("\n[bold blue]Available Components[/]")
    console.print("═" * 50)
    
    for name, info in components.items():
        console.print(f"\n[bold green]{name}[/]")
        console.print(f"  Description: {info['description']}")
        console.print(f"  Files: {', '.join(info['files'])}")
        if info['dependencies']:
            console.print(f"  Dependencies: {', '.join(info['dependencies'])}")
    
    console.print("\n" + "═" * 50)

def _show_component_diff(component: str, adding: bool = True):
    """Show diff for adding/removing a component"""
    components = {
        "testing": {
            "files": {
                "pytest.ini": """
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
addopts = -v --cov=. --cov-report=term-missing
""",
                "tests/test_together.py": """
import pytest
from unittest.mock import Mock, patch
from together_client import TogetherAIClient

@pytest.fixture
async def mock_client():
    with patch('together_client.TogetherAIClient') as mock:
        client = Mock()
        mock.return_value = client
        yield client

@pytest.mark.asyncio
async def test_embeddings(mock_client):
    texts = ["test text"]
    mock_client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    result = await mock_client.get_embeddings(texts)
    assert len(result) == 1
""",
                "requirements-test.txt": """
pytest==7.4.0
pytest-asyncio==0.21.1
pytest-cov==4.1.0
"""
            }
        },
        "logging": {
            "files": {
                "logging_config.py": """
import structlog
import logging

def configure_logging():
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
"""
            }
        }
    }
    
    if component not in components:
        console.print(f"[red]Error: Component '{component}' not found[/]")
        console.print("[yellow]Use 'compose list' to see available components[/]")
        return
    
    action = "[green]+" if adding else "[red]-"
    console.print(f"\n[bold]Changes for {action} {component}:[/]")
    console.print("═" * 50)
    
    for filename, content in components[component]["files"].items():
        console.print(f"\n[bold]{filename}[/]")
        console.print("─" * 50)
        
        for line in content.strip().split('\n'):
            prefix = "[green]+ " if adding else "[red]- "
            console.print(f"{prefix}{line}[/]")
    
    console.print("\n[bold]To apply these changes, run:[/]")
    if adding:
        console.print(f"[blue]  python -m pip install -r requirements-test.txt[/]" if component == "testing" else "")
        console.print(f"[blue]  codeai compose apply {component}[/]")
    else:
        console.print(f"[blue]  codeai compose remove {component} --confirm[/]")

if __name__ == '__main__':
    cli()