# cli.py
import click
import os
import asyncio
import datetime
import pickle
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
import yaml
from dotenv import load_dotenv
from analyzer import CodebaseAnalyzer

console = Console()

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
    
    config = {"together_api_key": api_key}
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    
    return api_key

@click.group()
def cli():
    """Code Understanding Assistant powered by Together AI"""
    pass

@cli.command()
def configure():
    """Configure API keys and settings"""
    try:
        api_key = setup_together_api()
        console.print("[green]Configuration saved successfully![/]")
    except Exception as e:
        console.print(f"[red]Error during configuration: {str(e)}[/]")
        exit(1)

@cli.command()
@click.argument('source')
@click.option('--github', is_flag=True, help='Treat source as GitHub URL')
@click.option('--project', '-p', help='Project name to store index in')
def init(source: str, github: bool, project: str = None):
    """Initialize and index a codebase"""
    try:
        # Ensure API key is configured
        api_key = setup_together_api()
        console.print(f"[green]Using API key:[/] {api_key[:5]}...{api_key[-5:]}")
        
        # Set up project if specified
        if project:
            create_project_cmd = get_command('create_project')
            ctx = click.Context(create_project_cmd, info_name='create_project')
            create_project_cmd.invoke(ctx, project_name=project)
        
        async def run_setup_and_indexing():
            try:
                # Initialize the analyzer with Together AI client
                if github:
                    console.print(f"[bold]Initializing from GitHub repository:[/] {source}")
                    analyzer = await CodebaseAnalyzer.from_github(source, api_key, project)
                else:
                    console.print(f"[bold]Initializing from local directory:[/] {source}")
                    analyzer = CodebaseAnalyzer(source, api_key, project)
                
                # Index the codebase
                await analyzer.index()
                
                # Save analyzer state
                console.print("[green]Saving analyzer state...[/]")
                save_analyzer_state(analyzer)
                console.print(f"[bold green]✓ Codebase indexed successfully in project '{analyzer.project_name}'![/]")
                
            except Exception as e:
                console.print(f"[bold red]Error in async operation: {str(e)}[/]")
                import traceback
                traceback.print_exc()
                raise
        
        # Run the async function
        console.print("[bold blue]Starting async operation...[/]")
        asyncio.run(run_setup_and_indexing())
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        import traceback
        traceback.print_exc()
        exit(1)

@cli.command()
@click.option('--interactive', '-i', is_flag=True, help='Start interactive mode')
@click.option('--chunks', '-c', type=int, help='Number of code chunks to use for context')
@click.option('--reset', '-r', is_flag=True, help='Reset conversation history before starting')
@click.option('--project', '-p', help='Project to use')
def ask(interactive: bool, chunks: int = None, reset: bool = False, project: str = None):
    """Ask questions about the codebase"""
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
                    response = await analyzer.query(question, chunks)
                console.print(f"\n[bold blue]Answer:[/] {response}\n")
                
                # Save analyzer state to persist conversation history
                save_analyzer_state(analyzer)
            except Exception as e:
                console.print(f"[bold red]Error generating response: {str(e)}[/]")
                import traceback
                traceback.print_exc()
        
        if interactive:
            console.print("[bold yellow]Interactive mode (Ctrl+C to exit)[/]")
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
        import traceback
        traceback.print_exc()
        exit(1)

@cli.command()
def reset_history():
    """Reset conversation history while keeping codebase index"""
    try:
        console.print("[bold blue]Loading code analyzer...[/]")
        analyzer = load_analyzer_state()
        
        # Reset conversation history
        analyzer.conversation_history = []
        save_analyzer_state(analyzer)
        console.print("[bold green]✓ Conversation history reset successfully![/]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        import traceback
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
    analyzer.load_state()
    return analyzer

def get_command(command_name):
    """Get a command from the CLI group by name"""
    return cli.get_command(None, command_name)

@cli.command()
@click.argument('project_name')
def create_project(project_name: str):
    """Create a new project or switch to an existing one"""
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
        import traceback
        traceback.print_exc()
        exit(1)

@cli.command()
def list_projects():
    """List all available projects"""
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
        import traceback
        traceback.print_exc()
        exit(1)

@cli.command()
def debug_projects():
    """Debug project registry (for development use)"""
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
            console.print(f"[dim]Entry: {name} = {info}[/]")
            if name != 'active':
                project_count += 1
                console.print(f"  - Project name: '{name}'")
                console.print(f"    Is active: {name == active}")
                console.print(f"    Created: {info.get('created', 'unknown')}")
        
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
        import traceback
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
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == '__main__':
    cli()