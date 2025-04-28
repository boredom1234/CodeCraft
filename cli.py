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
from analyzer import CodebaseAnalyzer, extract_code_structure
import time
from typing import Optional
import re

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
def configure():
    """Configure API keys and settings

    This command helps you set up the necessary API keys and configuration
    settings for the application. It will prompt you to enter your Together AI
    API key if it's not already set.
    """
    try:
        api_key = setup_together_api()
        console.print("[green]Configuration saved successfully![/]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/]")
        exit(0)
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        if os.getenv("DEBUG"):
            traceback.print_exc()
        exit(1)

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
@click.option('--workers', type=int, help='Number of parallel workers (default: CPU count)')
@click.option('--batch-size', type=int, default=10, help='Batch size for processing files')
@click.option('--distributed', is_flag=True, help='Enable distributed processing')
@click.option('--worker-addresses', help='Comma-separated list of worker addresses for distributed processing')
def init(source: str, github: bool, project: str = None, summary: bool = False, example: bool = False,
         workers: int = None, batch_size: int = 10, distributed: bool = False, worker_addresses: str = None):
    """Initialize and index a codebase.

    This command initializes a codebase for analysis. You can specify a local
    directory or a GitHub repository URL. Use the --github flag if the source
    is a GitHub URL. Optionally, specify a project name to store the index.

    Examples:
    
    # Initialize a local directory with parallel processing
    python cli.py init /path/to/codebase --workers 4 --batch-size 20

    # Initialize from a GitHub repository with distributed processing
    python cli.py init https://github.com/user/repo --github --distributed --worker-addresses "host1:5000,host2:5000"
    """
    try:
        # Show example usage if requested
        if example:
            console.print("[bold blue]Example Usage:[/]")
            console.print("\n[green]Initialize local directory with parallel processing:[/]")
            console.print("  python cli.py init /path/to/codebase --workers 4 --batch-size 20")
            console.print("\n[green]Initialize from GitHub with distributed processing:[/]")
            console.print("  python cli.py init https://github.com/user/repo --github --distributed --worker-addresses host1:5000,host2:5000")
            return

        # Ensure API key is configured
        api_key = setup_together_api()
        console.print(f"[green]Using API key:[/] {api_key[:5]}...{api_key[-5:]}")
        
        # Set up project if specified
        if project:
            create_project_cmd = get_command('create_project')
            ctx = click.Context(create_project_cmd, info_name='create_project')
            create_project(ctx, project_name=project)
        
        # Load config and update with CLI options
        config = load_config()
        if workers is not None:
            config['parallel'] = config.get('parallel', {})
            config['parallel']['max_workers'] = workers
        if batch_size is not None:
            config['parallel'] = config.get('parallel', {})
            config['parallel']['batch_size'] = batch_size
        if distributed:
            config['distributed'] = config.get('distributed', {})
            config['distributed']['enabled'] = True
            if worker_addresses:
                config['distributed']['workers'] = worker_addresses.split(',')
        
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
                analyzer.config.update(config)
                
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
@click.option('--concise', is_flag=True, help='Force concise responses (1-2 sentences)')
@click.option('--model', '-m', help='Specific model to use for this session')
def ask(interactive: bool, composer: bool = False, chunks: int = None, reset: bool = False, project: str = None, concise: bool = False, model: str = None):
    """Ask a question about the codebase.

    This command allows you to ask questions about your indexed codebase
    and get insightful answers based on semantic understanding of your code.

    Use interactive mode (-i) to maintain a conversation with the assistant.
    """
    try:
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
                # Handle some special queries directly
                if re.search(r'what(?:\'s| is) the project name', question, re.IGNORECASE):
                    name = analyzer.get_project_name()
                    print(f"\nThe project name is: {name}\n")
                    return
                
                with console.status("[bold yellow]Thinking...[/]"):
                    if composer:
                        # Get response in composer mode
                        response = await analyzer.query(
                            f"Act as a code composer. For the following request, show the exact changes needed using + for additions and - for removals: {question}",
                            chunks,
                            model=model
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
                        # Get response in normal mode, respecting concise flag
                        if concise:
                            # Add explicit concise instruction to question
                            question_with_instruction = f"Answer in 1-2 short sentences only: {question}"
                            response = await analyzer.query(question_with_instruction, chunks, model=model)
                        else:
                            response = await analyzer.query(question, chunks, model=model)
                
                # Display the response to the user
                console.print(f"\n{response}\n")
                
                # Save analyzer state to persist conversation history
                save_analyzer_state(analyzer)
                
            except Exception as e:
                console.print(f"[bold red]Error generating response: {str(e)}[/]")
                traceback.print_exc()
        
        if interactive:
            console.print("[bold yellow]Interactive mode (Ctrl+C to exit)[/]")
            if composer:
                console.print("[bold yellow]Composer mode enabled - showing changes as diffs[/]")
            if model:
                console.print(f"[bold cyan]Using model: {model}[/]")
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
    
    # If no project name is provided, try to get the active project
    if not project_name:
        try:
            registry_file = Path(".codeai") / "registry.pkl"
            if registry_file.exists():
                with open(registry_file, 'rb') as f:
                    projects = pickle.load(f)
                    project_name = projects.get('active')
                    console.print(f"[green]Using active project: {project_name}[/]")
        except Exception as e:
            console.print(f"[yellow]Could not determine active project: {str(e)}[/]")
    
    # If we still don't have a project name, it's an error
    if not project_name:
        console.print("[yellow]No project specified and no active project found.[/]")
        console.print("[yellow]Please create a project with 'python cli.py create-project YOUR_PROJECT_NAME'[/]")
        exit(1)
    
    # Create analyzer with saved state
    analyzer = CodebaseAnalyzer(".", api_key, project_name)  # Path doesn't matter for loading
    
    try:
        analyzer.load_state()
        return analyzer
    except FileNotFoundError:
        console.print(f"[yellow]No saved state found for project '{project_name}'. Please index a codebase first using 'python cli.py init'[/]")
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

@cli.command()
@click.option('--project', '-p', help='Project to refresh')
@click.option('--summary', is_flag=True, help='Generate detailed codebase summary after refresh')
def refresh(project: str = None, summary: bool = False):
    """Refresh the codebase index by updating only changed files.
    
    This command refreshes the codebase index by detecting and processing
    only files that have been added, modified, or deleted since the last
    indexing operation. This is much faster than re-indexing the entire codebase.
    
    Examples:
        codeai refresh
        codeai refresh --project myproject --summary
    """
    try:
        # Change to specified project if requested
        if project:
            create_project_cmd = get_command('create_project')
            ctx = click.Context(create_project_cmd, info_name='create_project')
            create_project_cmd.invoke(ctx, project_name=project)
            
        # Load the analyzer state
        console.print("[bold blue]Loading code analyzer...[/]")
        analyzer = load_analyzer_state(project)
        
        # Update config with summary option
        if summary:
            analyzer.config['generate_summary'] = True
            console.print("[bold yellow]Detailed codebase summary generation enabled[/]")
        
        async def run_refresh():
            try:
                # Refresh the index
                await analyzer.refresh_index()
                
                # Save analyzer state
                console.print("[green]Saving analyzer state...[/]")
                save_analyzer_state(analyzer)
                console.print(f"[bold green]✓ Codebase index refreshed successfully in project '{analyzer.project_name}'![/]")
                
            except Exception as e:
                console.print(f"[bold red]Error in async operation: {str(e)}[/]")
                traceback.print_exc()
                raise
        
        # Run the async function
        console.print("[bold blue]Starting refresh operation...[/]")
        asyncio.run(run_refresh())
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()
        exit(1)

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--line', '-l', type=int, help='Line number for context')
@click.option('--scope', '-s', is_flag=True, help='Show current scope information')
@click.option('--patterns', '-p', is_flag=True, help='Show similar code patterns')
def complete(file_path: str, line: int = None, scope: bool = False, patterns: bool = False):
    """Get AI-powered code completions with full codebase context.
    
    This command provides intelligent code completions based on:
    - The entire codebase context
    - Project patterns and coding style
    - Type information and imports
    - Current scope and dependencies
    
    Examples:
        # Get completion at specific line
        python cli.py complete path/to/file.py --line 42
        
        # Show scope and similar patterns
        python cli.py complete path/to/file.py --line 42 --scope --patterns
    """
    try:
        analyzer = load_analyzer_state()
        
        # Get the current line content if line number is provided
        current_line = ""
        if line:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if 0 <= line - 1 < len(lines):
                        current_line = lines[line - 1].strip()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                    if 0 <= line - 1 < len(lines):
                        current_line = lines[line - 1].strip()
                    console.print("[yellow]Warning: Some characters in the file could not be decoded properly.[/]")

        async def get_completion():
            # Show scope information if requested
            if scope:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        file_content = f.read()
                        console.print("[yellow]Warning: Some characters in the file could not be decoded properly.[/]")
                current_scope = analyzer._get_current_scope(file_content, line or 1)
                console.print("\n[bold blue]Current Scope:[/]")
                console.print(current_scope)
            
            # Show similar patterns if requested
            if patterns:
                query = f"Find code patterns similar to: {current_line}"
                relevant_chunks = await analyzer._get_relevant_chunks(query, 3)
                console.print("\n[bold blue]Similar Patterns:[/]")
                for chunk in relevant_chunks:
                    console.print(f"\n[dim]{chunk.metadata.get('file')}:[/]")
                    console.print(chunk.content.strip())
            
            # Get completion
            completion = await analyzer.complete_code(current_line, file_path, line)
            console.print("\n[bold green]Suggested completion:[/]")
            console.print(completion)

        asyncio.run(get_completion())
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--line', '-l', type=int, required=True, help='Line number to get suggestions for')
@click.option('--scope', '-s', is_flag=True, help='Show current scope information')
@click.option('--threshold', '-t', type=float, default=0.5, help='Minimum relevance score (0-1)')
def suggest(file_path: str, line: int, scope: bool = False, threshold: float = 0.5):
    """Get intelligent inline code suggestions with codebase awareness.
    
    This command provides multiple intelligent suggestions for improving
    or completing your code, considering:
    - Project-wide patterns and conventions
    - Current scope and context
    - Type safety and best practices
    - Similar code patterns from the codebase
    
    Each suggestion includes a relevance score based on:
    - Similarity to current code (20%)
    - Code style consistency (30%)
    - Semantic relevance to codebase (50%)
    
    Examples:
        # Get suggestions with default threshold
        python cli.py suggest path/to/file.py --line 42
        
        # Show scope and require high relevance
        python cli.py suggest path/to/file.py --line 42 --scope --threshold 0.8
    """
    try:
        analyzer = load_analyzer_state()
        
        # Get the current line content
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if 0 <= line - 1 < len(lines):
                current_line = lines[line - 1].strip()
            else:
                console.print("[red]Invalid line number[/]")
                return

        async def get_suggestions():
            # Show scope information if requested
            if scope:
                with open(file_path, 'r') as f:
                    file_content = f.read()
                current_scope = analyzer._get_current_scope(file_content, line)
                console.print("\n[bold blue]Current Scope:[/]")
                console.print(current_scope)
            
            # Get and filter suggestions
            suggestions = await analyzer.suggest_inline(file_path, line, current_line)
            suggestions = [s for s in suggestions if s['score'] >= threshold]
            
            if not suggestions:
                console.print(f"[yellow]No suggestions met the minimum relevance threshold of {threshold}[/]")
                return
            
            console.print("\n[bold green]Suggestions:[/]")
            for i, suggestion in enumerate(suggestions, 1):
                # Color based on score
                score = suggestion['score']
                if score >= 0.8:
                    color = "green"
                elif score >= 0.6:
                    color = "blue"
                else:
                    color = "yellow"
                
                # Show suggestion with context
                console.print(f"\n{i}. [{color}]{suggestion['text']}[/]")
                console.print(f"   [dim]Relevance: {score:.2%}[/]")
                
                # Show context if available
                if 'context' in suggestion:
                    ctx = suggestion['context']
                    if ctx.get('scope'):
                        console.print(f"   [dim]Scope: {ctx['scope']}[/]")

        asyncio.run(get_suggestions())
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--start-line', '-s', type=int, help='Start line number')
@click.option('--end-line', '-e', type=int, help='End line number')
@click.option('--detail', '-d', type=click.Choice(['low', 'medium', 'high']), default='medium', help='Level of detail in explanation')
def explain(file_path: str, start_line: int = None, end_line: int = None, detail: str = 'medium'):
    """Get natural language explanation of code.
    
    This command provides detailed explanations of code in natural language,
    focusing on key concepts and patterns.
    """
    try:
        analyzer = load_analyzer_state()
        
        # Read the relevant code section
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if start_line and end_line:
                if 0 <= start_line - 1 <= end_line - 1 < len(lines):
                    code = ''.join(lines[start_line - 1:end_line])
                else:
                    console.print("[red]Invalid line range[/]")
                    return
            else:
                code = ''.join(lines)

        async def get_explanation():
            explanation = await analyzer.explain_code(code, detail_level=detail)
            console.print("\n[bold green]Explanation:[/]")
            console.print(explanation)

        asyncio.run(get_explanation())
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()

@cli.command()
@click.argument('path', type=click.Path(exists=True), required=False)
@click.option('--watch', '-w', is_flag=True, help='Watch mode - continuously review code changes')
@click.option('--set-project-path', type=click.Path(exists=True), help='Set and save the project path for future use')
def assist(path: str = None, watch: bool = False, set_project_path: str = None):
    """Live code review assistant.
    
    Watches your codebase for changes and provides immediate feedback on:
    - Implementation errors
    - Logic flaws
    - Best practice violations
    - Potential bugs
    - Security issues
    
    Examples:
        # Set project path for future use
        python cli.py assist --set-project-path /path/to/your/project
        
        # Review using saved project path
        python cli.py assist
        
        # Review specific directory
        python cli.py assist /path/to/review
        
        # Watch mode for continuous review
        python cli.py assist --watch
    """
    try:
        analyzer = load_analyzer_state()
        config_dir = Path.home() / ".codeai"
        config_file = config_dir / "config.yml"
        
        # Load existing config
        config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        
        # Handle setting project path
        if set_project_path:
            config['project_path'] = str(Path(set_project_path).resolve())
            config_dir.mkdir(exist_ok=True)
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            console.print(f"[green]✓ Project path set to: {config['project_path']}[/]")
            if not path:  # If no path specified, use the newly set project path
                path = config['project_path']
        
        # If no path specified, try to use saved project path
        if not path:
            path = config.get('project_path', '.')
            if path == '.':
                console.print("[yellow]No project path specified or saved. Using current directory.[/]")
                console.print("[yellow]Tip: Use --set-project-path to save a default project path.[/]")
        
        # Define supported code file extensions
        CODE_EXTENSIONS = [
            '.py', '.js', '.ts', '.jsx', '.tsx',         # Python, JavaScript, TypeScript
            '.java', '.cpp', '.c', '.h', '.hpp', '.cs',   # Java, C++, C#
            '.go', '.rs',                                 # Go, Rust
            '.html', '.css', '.scss', '.sass',            # Web files
            '.vue', '.svelte',                            # Vue, Svelte
            '.json', '.xml', '.yaml', '.yml'              # Data files
        ]
        
        async def review_code(file_path: str) -> None:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Get static analysis issues
                issues = analyzer._analyze_code_issues(content, file_path)
                
                # Generate focused AI review
                prompt = f"""Review this code concisely. Focus ONLY on:
                1. Critical bugs or errors
                2. Major performance issues
                3. Security vulnerabilities
                4. Significant design flaws

                Format: "<file>:<line> - <type>: <brief description>"
                Keep each issue to one line. Skip minor style issues.

                Code to review:
                {content}
                """
                
                review = await analyzer.ai_client.get_completion(prompt)
                
                # Clear screen in watch mode
                if watch:
                    console.clear()
                    console.print(f"[bold blue]Changes in {Path(file_path).name}:[/]\n")
                
                # Print critical static analysis issues
                critical_issues = [i for i in issues if i["type"] in ("Error", "Warning")]
                if critical_issues:
                    for issue in critical_issues:
                        color = "red" if issue["type"] == "Error" else "yellow"
                        console.print(f"[{color}]{issue['file']}:{issue['line']} - {issue['description']}[/]")
                    console.print()
                
                # Print AI review
                console.print(review.strip())
                
            except Exception as e:
                console.print(f"[bold red]Error reviewing {file_path}: {str(e)}[/]")
        
        if watch:
            try:
                last_mtimes = {}
                console.print(f"[bold blue]Watching directory: {path}[/]")
                console.print("[dim]Press Ctrl+C to stop[/]\n")
                
                while True:
                    # Get all code files in the directory with supported extensions
                    all_files = []
                    for ext in CODE_EXTENSIONS:
                        all_files.extend(list(Path(path).rglob(f'*{ext}')))
                    
                    for file_path in all_files:
                        try:
                            mtime = os.path.getmtime(str(file_path))
                            if str(file_path) not in last_mtimes or mtime != last_mtimes[str(file_path)]:
                                last_mtimes[str(file_path)] = mtime
                                asyncio.run(review_code(str(file_path)))
                        except FileNotFoundError:
                            continue
                            
                    time.sleep(1)  # Check every second
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped watching[/]")
        else:
            # Review all code files with supported extensions
            all_files = []
            for ext in CODE_EXTENSIONS:
                all_files.extend(list(Path(path).rglob(f'*{ext}')))
            
            if not all_files:
                console.print(f"[yellow]No supported code files found in {path}[/]")
                console.print(f"[dim]Supported extensions: {', '.join(CODE_EXTENSIONS)}[/]")
                return
                
            console.print(f"[bold blue]Reviewing {len(all_files)} code files in {path}...[/]\n")
            for file_path in all_files:
                console.print(f"[dim]Reviewing {file_path}...[/]")
                asyncio.run(review_code(str(file_path)))
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()

def _detect_active_file() -> Optional[str]:
    """Try to detect which file the user is currently working on."""
    try:
        # List of supported code file extensions
        CODE_EXTENSIONS = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.cs', '.go', '.rs']
        
        current_dir = Path('.')
        code_files = []
        
        # Find all supported code files
        for ext in CODE_EXTENSIONS:
            code_files.extend(list(current_dir.rglob(f'*{ext}')))
        
        if not code_files:
            return None
            
        # Sort by modification time, most recent first
        code_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        return str(code_files[0])
        
    except Exception:
        return None

@cli.command()
def project_name():
    """Display the current project name."""
    try:
        # Load analyzer state
        analyzer = load_analyzer_state()
        if not analyzer:
            console.print("[yellow]No active project found. Please create or switch to a project first.[/]")
            return
        
        # Get project name
        name = analyzer.get_project_name()
        console.print(f"[bold green]Current project name:[/] {name}")
    except Exception as e:
        console.print(f"[bold red]Error getting project name: {str(e)}[/]")
        if os.getenv("DEBUG"):
            traceback.print_exc()

@cli.command()
@click.argument('model_name', required=False)
@click.option('--list', '-l', is_flag=True, help='List available models')
def set_model(model_name: str = None, list: bool = False):
    """Set the default model for code analysis.

    This command allows you to set the default language model to use for code analysis.
    If no model is specified, it shows the current default model.

    Examples:
        codeai set-model meta-llama/Llama-3.3-70B-Instruct-Turbo
        codeai set-model --list

    Available models include:
    - meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8  
    - meta-llama/Llama-4-Scout-17B-16E-Instruct
    - meta-llama/Llama-3.3-70B-Instruct-Turbo
    - meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
    - meta-llama/Llama-3.2-3B-Instruct-Turbo
    - deepseek-ai/DeepSeek-R1
    - deepseek-ai/DeepSeek-V3
    - deepseek-ai/deepseek-llm-67b-chat
    - google/gemma-3-27b-it
    - google/gemma-3-12b-it
    - mistralai/Mistral-Small-24B-Instruct-2501
    - mistralai/Mixtral-8x7B-Instruct-v0.1
    And many others. Use --list to see all available models.
    """
    try:
        # Define model categories for neat display
        model_categories = {
            "Meta LLaMA Series": [
                "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                "meta-llama/Llama-3.2-3B-Instruct-Turbo",
                "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            ],
            "DeepSeek AI Series": [
                "deepseek-ai/DeepSeek-R1",
                "deepseek-ai/DeepSeek-V3",
                "deepseek-ai/deepseek-llm-67b-chat",
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            ],
            "DeepCogito Series": [
                "deepcogito/cogito-v1-preview-llama-70B",
                "deepcogito/cogito-v1-preview-llama-8B",
                "deepcogito/cogito-v1-preview-llama-3B",
                "deepcogito/cogito-v1-preview-qwen-32B",
                "deepcogito/cogito-v1-preview-qwen-14B",
            ],
            "Google Gemma": [
                "google/gemma-3-27b-it",
                "google/gemma-3-12b-it",
                "google/gemma-3-1b-it",
                "google/gemma-2-27b-it",
                "google/gemma-2-9b-it",
                "google/gemma-2b-it",
            ],
            "Arcee AI": [
                "arcee-ai/virtuoso-large",
                "arcee-ai/virtuoso-medium",
                "arcee-ai/maestro-reasoning",
                "arcee-ai/coder-large",
                "arcee-ai/caller",
                "arcee-ai/arcee-blitz",
            ],
            "Qwen": [
                "Qwen/QwQ-32B",
                "Qwen/Qwen2.5-72B-Instruct-Turbo",
                "Qwen/Qwen2.5-7B-Instruct-Turbo",
                "Qwen/Qwen2.5-Coder-32B-Instruct",
            ],
            "Mistral AI": [
                "mistralai/Mistral-Small-24B-Instruct-2501",
                "mistralai/Mistral-7B-Instruct-v0.1",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
            ],
            "NVIDIA": [
                "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
            ],
            "Databricks": [
                "databricks/dbrx-instruct",
            ]
        }
        
        # Flatten the model list for checking
        all_models = [model for category in model_categories.values() for model in category]
        
        if list:
            # Show all available models in categories
            console.print("[bold blue]Available Models:[/]")
            for category, models in model_categories.items():
                console.print(f"\n[bold cyan]{category}[/]")
                for model in models:
                    console.print(f"  • {model}")
            return
        
        # Load config
        config_path = Path('config.yml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
            
        # Initialize model section if it doesn't exist
        if 'model' not in config:
            config['model'] = {}
            
        # Show current model if no new model specified
        if not model_name:
            current_model = config.get('model', {}).get('default', "meta-llama/Llama-3.3-70B-Instruct-Turbo")
            console.print(f"[bold blue]Current default model:[/] {current_model}")
            console.print("\nUse 'set-model --list' to see all available models.")
            return
            
        # Check if valid model
        if model_name not in all_models:
            console.print(f"[bold red]Warning:[/] '{model_name}' is not in the list of common models.")
            if not Confirm.ask("[bold yellow]Continue with this model anyway?[/]"):
                console.print("[yellow]Model change cancelled.[/]")
                return
            
        # Update config
        config['model']['default'] = model_name
        
        # Save config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        console.print(f"[bold green]✓ Default model updated to:[/] {model_name}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        traceback.print_exc()
        exit(1)

if __name__ == '__main__':
    cli()