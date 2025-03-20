# analyzer.py
from typing import List, Dict, Optional, Tuple
import faiss
import numpy as np
from pathlib import Path
import pickle
import os
import traceback
from together_client import TogetherAIClient
from git import Repo
from rich.progress import Progress, TextColumn, BarColumn, TaskID, TimeRemainingColumn, SpinnerColumn, TimeElapsedColumn, TaskProgressColumn
from rich.console import Console
import re
import yaml
import ast
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.style import Style
from rich.text import Text
from rich.columns import Columns
from rich.console import Group
import datetime
from collections import defaultdict

console = Console()

class CodeStructure:
    """Helper class to track code structure and context.

    Attributes:
        imports (List[str]): A list of import statements found in the code.
        classes (Dict[str, Dict[str, List[str]]]): A dictionary mapping class names to their methods and attributes.
        functions (List[str]): A list of function names found in the code.
        global_vars (List[str]): A list of global variables found in the code.
        docstrings (Dict[ast.AST, str]): A dictionary mapping AST nodes to their docstrings.
        comments (List[str]): A list of comments found in the code.
    """
    
    def __init__(self):
        """Initialize a new CodeStructure instance with empty lists and dictionaries."""
        self.imports = []
        self.classes = {}  # class_name -> {methods: [], attributes: []}
        self.functions = []
        self.global_vars = []
        self.docstrings = {}  # node -> docstring
        self.comments = []
        
    def add_import(self, import_stmt: str):
        """Add an import statement to the structure.

        Args:
            import_stmt (str): The import statement to add.
        """
        self.imports.append(import_stmt.strip())
        
    def add_class(self, class_name: str, methods: List[str], attributes: List[str]):
        """Add a class with its methods and attributes to the structure.

        Args:
            class_name (str): The name of the class.
            methods (List[str]): A list of method names in the class.
            attributes (List[str]): A list of attribute names in the class.
        """
        self.classes[class_name] = {"methods": methods, "attributes": attributes}
        
    def add_function(self, func_name: str):
        """Add a function name to the structure.

        Args:
            func_name (str): The name of the function.
        """
        self.functions.append(func_name)
        
    def add_global(self, var_name: str):
        """Add a global variable name to the structure.

        Args:
            var_name (str): The name of the global variable.
        """
        self.global_vars.append(var_name)
        
    def add_docstring(self, node: ast.AST, docstring: str):
        """Add a docstring associated with an AST node to the structure.

        Args:
            node (ast.AST): The AST node associated with the docstring.
            docstring (str): The docstring text.
        """
        self.docstrings[node] = docstring.strip()
        
    def add_comment(self, comment: str):
        """Add a comment to the structure.

        Args:
            comment (str): The comment text.
        """
        self.comments.append(comment.strip())

def extract_code_structure(content: str, file_type: str = 'py') -> CodeStructure:
    """Extract structural information from code with language-specific parsing"""
    structure = CodeStructure()
    
    try:
        if file_type == 'py':
            # Python-specific parsing using ast
            tree = ast.parse(content)
            
            # Extract docstrings and comments
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        structure.add_docstring(node, docstring)
            
            # Process each node in the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        structure.add_import(name.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for name in node.names:
                        import_stmt = f"from {module} import {name.name}"
                        structure.add_import(import_stmt)
                elif isinstance(node, ast.ClassDef):
                    methods = []
                    attributes = []
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    attributes.append(target.id)
                    
                    structure.add_class(node.name, methods, attributes)
                elif isinstance(node, ast.FunctionDef):
                    structure.add_function(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            structure.add_global(target.id)
        
        elif file_type in ['go', 'js', 'ts', 'jsx', 'tsx']:
            # Generic parsing for other languages using regex
            # Extract imports
            import_patterns = {
                'go': r'import\s+(?:\(\s*|\s*)([^)]+)',
                'js': r'(?:import\s+.*?from\s+[\'"].*?[\'"]|require\s*\([\'"].*?[\'"]\))',
                'ts': r'import\s+.*?from\s+[\'"].*?[\'"]',
            }
            
            if file_type in import_patterns:
                pattern = import_patterns[file_type]
                imports = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                for imp in imports:
                    structure.add_import(imp.strip())
            
            # Extract functions
            func_patterns = {
                'go': r'func\s+(\w+)',
                'js': r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?function)',
                'ts': r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?function)',
            }
            
            if file_type in func_patterns:
                pattern = func_patterns[file_type]
                functions = re.findall(pattern, content)
                for func in functions:
                    if isinstance(func, tuple):
                        func = next((f for f in func if f), '')
                    structure.add_function(func.strip())
            
            # Extract classes
            class_patterns = {
                'go': r'type\s+(\w+)\s+struct',
                'js': r'class\s+(\w+)',
                'ts': r'class\s+(\w+)',
            }
            
            if file_type in class_patterns:
                pattern = class_patterns[file_type]
                classes = re.findall(pattern, content)
                for class_name in classes:
                    # Try to find methods within class scope
                    if file_type == 'go':
                        method_pattern = rf'func\s+\(\w+\s+\*?{class_name}\)\s+(\w+)'
                    else:
                        method_pattern = rf'(?:async\s+)?(\w+)\s*\([^)]*\)\s*{{[^}}]*}}'
                    
                    methods = re.findall(method_pattern, content)
                    structure.add_class(class_name, methods, [])
        
        # Extract comments (works for most languages)
        comment_patterns = [
            r'//.*$',           # Single line comments
            r'/\*[\s\S]*?\*/',  # Multi-line comments
            r'#.*$',            # Python/Shell style comments
        ]
        
        for pattern in comment_patterns:
            comments = re.findall(pattern, content, re.MULTILINE)
            for comment in comments:
                structure.add_comment(comment.strip())
        
        return structure
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not fully parse code structure: {str(e)}[/]")
        return structure

def get_context_window(content: List[str], current_line: int, window_size: int = 3) -> List[str]:
    """Get surrounding context lines"""
    start = max(0, current_line - window_size)
    end = min(len(content), current_line + window_size + 1)
    return content[start:end]

def format_chunk_with_context(chunk_lines: List[str], structure: CodeStructure, 
                            start_line: int, file_path: str) -> str:
    """Format a chunk with its structural context using markdown-style formatting"""
    # Start with file information as a header
    lines = [f"# {file_path}\n"]
    
    # Add relevant imports in a code block
    if structure.imports:
        lines.append("## Imports\n```python")
        lines.extend(structure.imports[:5])  # Show first 5 imports
        lines.append("```\n")
        
    # Add class context if chunk is part of a class
    class_context = []
    for class_name, details in structure.classes.items():
        if any(class_name in line for line in chunk_lines):
            class_context.append(f"## Class `{class_name}`")
            if details['methods']:
                class_context.append("### Methods")
                methods = [f"`{m}`" for m in details['methods']]
                class_context.append(", ".join(methods))
            if details['attributes']:
                class_context.append("### Attributes")
                attrs = [f"`{a}`" for a in details['attributes']]
                class_context.append(", ".join(attrs))
            class_context.append("")  # Add blank line
    lines.extend(class_context)
    
    # Add function context
    func_context = []
    functions = [func_name for func_name in structure.functions 
                if any(func_name in line for line in chunk_lines)]
    if functions:
        func_context.append("## Functions")
        funcs = [f"`{f}`" for f in functions]
        func_context.append(", ".join(funcs))
        func_context.append("")  # Add blank line
    lines.extend(func_context)
    
    # Add relevant docstrings in blockquotes
    docstrings = [doc for doc in structure.docstrings.values() if any(
        line.strip() in ''.join(chunk_lines) for line in doc.split('\n')
    )]
    if docstrings:
        lines.append("## Documentation")
        for doc in docstrings[:2]:  # Show first 2 relevant docstrings
            # Format docstring as markdown blockquote
            doc_lines = doc.split('\n')
            formatted_doc = '\n'.join(f"> {line}" if line.strip() else ">" for line in doc_lines)
            lines.append(formatted_doc)
        lines.append("")  # Add blank line
    
    # Add the actual code chunk with line numbers in a code block
    lines.append("## Code\n```python")
    # Add a header row for line numbers
    lines.append("# Line | Code")
    lines.append("# ---- | ----")
    for i, line in enumerate(chunk_lines, start=start_line):
        # Escape any backticks in the code to prevent markdown formatting issues
        escaped_line = line.rstrip().replace("`", "\\`")
        # Right-align line numbers and add a monospace format
        lines.append(f"# {i:4d} | {escaped_line}")
    lines.append("```")
    
    # Add any inline comments as a separate section
    comments = [comment for comment in structure.comments 
               if any(comment in line for line in chunk_lines)]
    if comments:
        lines.append("\n## Comments")
        for comment in comments:
            lines.append(f"* {comment}")
    
    return "\n".join(lines)

class SemanticChunk:
    """A semantic chunk of code with enhanced context information."""
    def __init__(self, content: str, metadata: Dict):
        self.content = content
        self.metadata = metadata
        self.embedding = None
        self.importance_score = 0.0
        
    def add_context(self, context_type: str, context: str):
        """Add additional context to the chunk."""
        if 'contexts' not in self.metadata:
            self.metadata['contexts'] = {}
        self.metadata['contexts'][context_type] = context
        
    def calculate_importance(self):
        """Calculate the importance score of this chunk."""
        # Base score
        score = 1.0
        
        # Adjust based on content type
        if self.metadata.get('type') == 'class_definition':
            score *= 1.5
        elif self.metadata.get('type') == 'function_definition':
            score *= 1.3
        elif self.metadata.get('type') == 'import_section':
            score *= 1.2
            
        # Adjust based on documentation
        if self.metadata.get('has_docstring'):
            score *= 1.2
        if self.metadata.get('has_comments'):
            score *= 1.1
            
        # Adjust based on complexity
        complexity = self.metadata.get('complexity', 0)
        score *= (1 + (complexity * 0.1))
        
        self.importance_score = score

class CodebaseAnalyzer:
    def __init__(self, path: str, together_api_key: str, project_name: str = None):
        self.path = Path(path)
        self.ai_client = TogetherAIClient(together_api_key)
        self.faiss_index = None
        self.documents = []
        self.metadata = []
        self.conversation_history = []
        
        # Initialize Rich console with markdown support and force color
        self.console = Console(force_terminal=True, color_system="truecolor")
        
        # Initialize config with defaults
        self.config = {
            "chunk_size": 20,
            "overlap": 5,
            "max_history": 5,
            "temperature": 0.7,
            "debug": False,
            "generate_summary": False
        }
        
        # Create data directory if it doesn't exist
        self.data_dir = Path(".codeai")
        self.data_dir.mkdir(exist_ok=True)
        
        # Set up project-specific data directory
        if project_name:
            self.project_name = project_name
            self.project_dir = self.data_dir / "projects" / project_name
            self.project_dir.mkdir(exist_ok=True, parents=True)
        else:
            # Try to get active project from registry
            registry_file = self.data_dir / "registry.pkl"
            if registry_file.exists():
                with open(registry_file, 'rb') as f:
                    projects = pickle.load(f)
                    active = projects.get('active', None)
                    if active:
                        self.project_name = active
                        self.project_dir = self.data_dir / "projects" / active
                        self.project_dir.mkdir(exist_ok=True, parents=True)
                    else:
                        self.project_name = "default"
                        self.project_dir = self.data_dir
            else:
                self.project_name = "default"
                self.project_dir = self.data_dir
                
        # Load project config if it exists
        config_file = self.project_dir / "config.yml"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    project_config = yaml.safe_load(f)
                    if isinstance(project_config, dict):
                        self.config.update(project_config)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load project config: {str(e)}[/]")
        
        # Create a layout for better organization
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header"),
            Layout(name="main", ratio=8),
            Layout(name="footer")
        )
    
    @classmethod
    async def from_github(cls, github_url: str, api_key: str, project_name: str = None):
        """Create analyzer from GitHub repository asynchronously"""
        console.print(f"[bold blue]Cloning GitHub repository:[/] {github_url}")
        
        # Use project directory if specified
        if project_name:
            data_dir = Path(".codeai") / "projects" / project_name
        else:
            data_dir = Path(".codeai")
        
        data_dir.mkdir(exist_ok=True, parents=True)
        
        # Clone to a temporary directory
        repo_name = github_url.split('/')[-1]
        clone_path = data_dir / "repos" / repo_name
        clone_path.parent.mkdir(exist_ok=True)
        
        try:
            if not clone_path.exists():
                # Create a new instance of the analyzer
                analyzer = cls(str(clone_path), api_key, project_name)
                
                # Clone the repository synchronously (git operations are not async)
                console.print(f"[blue]Cloning to:[/] {clone_path}")
                with console.status("[bold green]Cloning repository..."):
                    Repo.clone_from(github_url, clone_path)
                
                # Check if the clone worked
                files = list(clone_path.glob("*"))
                console.print(f"[green]Clone successful. Found {len(files)} items in repository.[/]")
                if files:
                    console.print(f"[dim]Top-level items:[/] {', '.join([f.name for f in files[:5]])}{' ...' if len(files) > 5 else ''}")
                
                return analyzer
            else:
                # If already cloned, check the directory content
                files = list(clone_path.glob("*"))
                console.print(f"[yellow]Repository already exists at {clone_path}[/]")
                console.print(f"[green]Found {len(files)} items in repository.[/]")
                if files:
                    console.print(f"[dim]Top-level items:[/] {', '.join([f.name for f in files[:5]])}{' ...' if len(files) > 5 else ''}")
                
                # If already cloned, just return the analyzer
                return cls(str(clone_path), api_key, project_name)
        except Exception as e:
            console.print(f"[bold red]Error cloning repository:[/] {str(e)}")
            traceback.print_exc()
            raise
    
    async def index(self):
        """Index the codebase with detailed progress feedback."""
        console.print("[bold blue]Starting indexing process...[/]")
        
        try:
            # Create a rich progress display with more detailed feedback
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}[/]"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                # Find all files
                task_find = progress.add_task("[green]Finding files...", total=None)
                all_files = list(self.path.rglob("*.*"))
                progress.update(task_find, total=len(all_files), completed=len(all_files))
                console.print(f"[green]Found {len(all_files)} files in total[/]")
                
                # Show sample of files for debugging
                if all_files:
                    sample_files = all_files[:5]
                    console.print(f"[dim]Sample files: {', '.join([str(f) for f in sample_files[:5]])}{' ...' if len(all_files) > 5 else ''}[/]")
                
                # Filter indexable files - with more detailed debugging
                task_filter = progress.add_task("[green]Filtering indexable files...", total=len(all_files))
                indexable_files = []
                file_extensions = {}  # Track extension counts
                
                for f in all_files:
                    progress.update(task_filter, advance=1)
                    if self._should_index_file(f):
                        indexable_files.append(f)
                        ext = f.suffix.lower()
                        file_extensions[ext] = file_extensions.get(ext, 0) + 1
                
                # Print detailed extension breakdown
                progress.update(task_filter, completed=len(all_files))
                console.print(f"[green]Found {len(indexable_files)} indexable files[/]")
                if file_extensions:
                    console.print("[green]Extension breakdown:[/]")
                    for ext, count in sorted(file_extensions.items(), key=lambda x: x[1], reverse=True):
                        console.print(f"  [blue]{ext}:[/] {count} files")
                else:
                    console.print("[yellow]No valid file extensions found[/]")
                
                # Additional debugging for file paths
                if not indexable_files:
                    console.print("[yellow]All files were filtered out. Checking file paths:[/]")
                    all_paths = set()
                    for file in all_files:
                        file_path = str(file)
                        parts = file_path.split(os.sep)
                        for part in parts:
                            all_paths.add(part)
                    console.print(f"[dim]Unique path components: {sorted(all_paths)}[/]")
                    
                    console.print("[bold red]Error: No indexable files found in the repository[/]")
                    raise ValueError("No indexable files found in the repository")
                
                # Process files
                all_embeddings = []
                all_documents = []
                all_metadata = []
                
                # Create tasks for file processing and embedding generation
                task_process = progress.add_task("[green]Processing files...", total=len(indexable_files))
                task_embed = progress.add_task("[yellow]Generating embeddings...", total=len(indexable_files), visible=False)
                
                processed_files = 0
                embedded_files = 0
                error_files = 0
                
                # Process files one by one for better debugging
                for file_idx, file_path in enumerate(indexable_files):
                    try:
                        progress.update(task_process, advance=1)
                        console.print(f"[dim][{file_idx+1}/{len(indexable_files)}] Processing {file_path}[/]")
                        
                        chunks = self._chunk_file(file_path)
                        if not chunks:
                            console.print(f"[yellow]Warning: No chunks were created for {file_path}[/]")
                            continue
                            
                        # Get the content from each chunk
                        texts = [chunk.content for chunk in chunks]  # Changed from chunk["text"] to chunk.content
                        
                        # Get embeddings
                        try:
                            progress.update(task_embed, visible=True)
                            embeddings = await self.ai_client.get_embeddings(texts)
                            progress.update(task_embed, advance=1)
                            
                            if not embeddings:
                                console.print(f"[yellow]Warning: No embeddings were returned for {file_path}[/]")
                                continue
                                
                            all_embeddings.extend(embeddings)
                            all_documents.extend(texts)
                            all_metadata.extend([{
                                "file": str(file_path.relative_to(self.path)),
                                "start_line": chunk.metadata['start_line'],  # Changed from chunk["start_line"]
                                "end_line": chunk.metadata['end_line']  # Changed from chunk["end_line"]
                            } for chunk in chunks])
                            
                            embedded_files += 1
                            console.print(f"[green]✓ Successfully embedded {file_path} ({len(chunks)} chunks)[/]")
                        except Exception as e:
                            error_files += 1
                            console.print(f"[red]Error generating embeddings for {file_path}: {str(e)}[/]")
                            traceback.print_exc()
                    except Exception as e:
                        error_files += 1
                        console.print(f"[red]Error processing {file_path}: {str(e)}[/]")
                        traceback.print_exc()
                    
                    processed_files += 1
                
                if not all_embeddings:
                    console.print("[bold red]Error: No embeddings were generated. Check file processing.[/]")
                    raise ValueError("No embeddings were generated. Check file processing.")
                
                # Create FAISS index
                task_index = progress.add_task("[blue]Creating FAISS index...", total=1)
                console.print(f"[blue]Creating FAISS index with {len(all_embeddings)} embeddings[/]")
                
                try:
                    dimension = len(all_embeddings[0])
                    self.faiss_index = faiss.IndexFlatL2(dimension)
                    embeddings_array = np.array(all_embeddings).astype('float32')
                    self.faiss_index.add(embeddings_array)
                    
                    # Store documents and metadata
                    self.documents = all_documents
                    self.metadata = all_metadata
                    
                    progress.update(task_index, advance=1)
                    
                    # Generate codebase summary only if requested in config
                    if self.config.get('generate_summary', False):
                        task_summary = progress.add_task("[blue]Generating codebase summary...", total=1)
                        console.print("[blue]Generating high-level codebase summary...[/]")
                        await self._generate_codebase_summary(indexable_files)
                        progress.update(task_summary, advance=1)
                    else:
                        # Just create a minimal summary
                        self.codebase_summary = f"# Codebase Summary\n\nTotal files: {len(indexable_files)}\nFile types: {', '.join(f'{ext} ({count})' for ext, count in sorted(file_extensions.items(), key=lambda x: x[1], reverse=True))}"
                    
                    # Save state
                    task_save = progress.add_task("[blue]Saving state...", total=1)
                    console.print("[blue]Saving state...[/]")
                    self.save_state()
                    progress.update(task_save, advance=1)
                    
                    console.print(f"[bold green]Indexing complete! Processed {processed_files} files, embedded {embedded_files} files, encountered {error_files} errors.[/]")
                except Exception as e:
                    console.print(f"[bold red]Error creating FAISS index: {str(e)}[/]")
                    traceback.print_exc()
                    raise
                
        except Exception as e:
            console.print(f"[bold red]Indexing failed: {str(e)}[/]")
            traceback.print_exc()
            raise
    
    def _create_header(self, title: str) -> Panel:
        """Create a styled header panel"""
        from rich.text import Text
        from rich import box
        
        # Create a more professional header with proper text styling and alignment
        header_text = Text(title, style="bold white on blue", justify="center")
        
        return Panel(
            header_text,
            box=box.HEAVY_HEAD,
            border_style="blue",
            padding=(1, 2)
        )

    def _create_footer(self, text: str) -> Panel:
        """Create a styled footer panel"""
        from rich.text import Text
        from rich import box
        
        # Create a professional footer with metadata
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer_content = Text(f"{text} | Generated: {timestamp}", style="dim", justify="center")
        
        return Panel(
            footer_content,
            box=box.HORIZONTALS,
            border_style="blue",
            padding=(1, 1)
        )

    def _format_output(self, text: str, title: str = "Code Analysis Report"):
        """Format text using Rich's markdown renderer with enhanced layout"""
        try:
            # Import required components for rich formatting
            from rich.markdown import Markdown
            from rich.panel import Panel
            from rich import box
            from rich.table import Table
            
            # Create header
            header = self._create_header(title)
            
            # Process the text to enhance code blocks
            enhanced_text = self._enhance_code_blocks(text)
            
            # Create main content with professional styling
            md = Markdown(enhanced_text)
            main_panel = Panel(
                md,
                border_style="blue",
                box=box.HEAVY,
                padding=(1, 2),
                title="Technical Analysis",
                title_align="left"
            )
            
            # Create footer with stats
            stats_table = Table.grid(padding=1)
            stats_table.add_column(style="bold blue", justify="right")
            stats_table.add_column(style="white")
            
            stats_table.add_row("Project:", self.project_name)
            stats_table.add_row("Indexed Files:", str(len(set(m['file'] for m in self.metadata))))
            stats_table.add_row("Total Chunks:", str(len(self.documents)))
            stats_table.add_row("History Items:", str(len(self.conversation_history)))
            
            footer = self._create_footer("AI Code Analysis Engine")
            
            # Update layout
            self.layout["header"].update(header)
            self.layout["main"].update(main_panel)
            self.layout["footer"].update(footer)
            
            # Print the layout
            self.console.print(self.layout)
            
        except Exception as e:
            self.console.print(f"[bold red]Error formatting output:[/] {str(e)}")
            # Fallback to simple formatting
            self.console.print(f"\n[bold blue]{title}[/]\n\n{text}\n")

    async def query(self, question: str, chunk_count: int = None) -> str:
        """Query the codebase with enhanced semantic understanding."""
        try:
            # Analyze query intent and context
            query_context = self._analyze_query(question)
            
            # Get question embedding
            question_embedding = await self.ai_client.get_embeddings([question])
            question_vector = np.array(question_embedding).astype('float32')

            # Determine optimal chunk count based on query complexity
            if not chunk_count:
                chunk_count = self._determine_optimal_chunk_count(question, query_context)

            # Get initial semantic matches
            D, I = self.faiss_index.search(question_vector, chunk_count * 2)  # Get more candidates
            
            # Display the search process to the user
            self.console.print(f"[bold blue]Searching for relevant code to answer:[/] {question}")
            self.console.print(f"[dim]Using {chunk_count} chunks for context[/]")

            # Score and rank chunks based on multiple factors
            scored_chunks = []
            for idx in I[0]:
                if idx < len(self.documents):
                    chunk = self.documents[idx]
                    
                    # Handle both SemanticChunk objects and raw document strings
                    if isinstance(chunk, SemanticChunk):
                        # Get the embedding distance
                        distance = D[0][list(I[0]).index(idx)]
                        
                        # Convert distance to similarity score (1/(1+distance) gives a value between 0-1)
                        similarity_score = 1.0 / (1.0 + distance)
                        
                        # Use existing relevance calculation for SemanticChunk
                        relevance_score = self._calculate_chunk_relevance(
                            chunk,
                            query_context,
                            distance
                        )
                        
                        # Store the scores in the chunk metadata for display
                        chunk.metadata['similarity_score'] = similarity_score
                        chunk.metadata['relevance_score'] = relevance_score
                        
                        scored_chunks.append((relevance_score, chunk))
                    else:
                        # For raw document strings, use a simpler relevance calculation
                        # based just on the embedding distance
                        distance = D[0][list(I[0]).index(idx)]
                        similarity_score = 1.0 / (1.0 + distance)
                        relevance_score = similarity_score
                        
                        # Create a simple metadata dict from the corresponding metadata
                        metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                        metadata['similarity_score'] = similarity_score
                        metadata['relevance_score'] = relevance_score
                        
                        # Create a temporary SemanticChunk for consistency
                        temp_chunk = SemanticChunk(chunk, metadata)
                        scored_chunks.append((relevance_score, temp_chunk))

            # Sort by relevance and take top chunks
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            
            # Store the relevance scores in the chunk metadata before extracting just the chunks
            for score, chunk in scored_chunks[:chunk_count]:
                chunk.metadata['similarity_score'] = score  # Make sure the score is saved in metadata
            
            selected_chunks = [chunk for _, chunk in scored_chunks[:chunk_count]]

            self.console.print(f"[bold green]Found {len(selected_chunks)} relevant code sections[/]")

            # Build context from chunks
            if selected_chunks:
                if isinstance(selected_chunks[0], SemanticChunk):
                    # Use enhanced context building for SemanticChunk objects
                    context = self._build_enhanced_context(selected_chunks, query_context)
                else:
                    # Fallback for raw document strings
                    context = self._build_simple_context(selected_chunks, query_context)
            else:
                context = "No relevant code found in the codebase."

            # Include conversation history if relevant
            history_context = ""
            if self.conversation_history:
                history_context = self._prepare_filtered_history(question, query_context)

            # Generate response with enhanced context
            response = await self._generate_enhanced_response(
                question,
                context,
                history_context,
                query_context
            )

            # Update conversation history
            self._update_conversation_history(question, response, query_context)

            return response

        except Exception as e:
            console.print(f"[bold red]Error generating response: {str(e)}[/]")
            traceback.print_exc()
            raise
    
    def _calculate_chunk_relevance(self, chunk: SemanticChunk, query_context: Dict, embedding_distance: float) -> float:
        """Calculate chunk relevance score using multiple factors."""
        # Start with embedding similarity (convert distance to similarity)
        score = 1.0 / (1.0 + embedding_distance)
        
        # Boost by chunk importance
        score *= (1.0 + chunk.importance_score * 0.5)
        
        # Context match boost
        if query_context['type'] == 'explain' and chunk.metadata.get('has_docstring'):
            score *= 1.3
        elif query_context['type'] == 'modify' and chunk.metadata.get('type') in ['class_definition', 'function_definition']:
            score *= 1.2
        elif query_context['type'] == 'error' and chunk.metadata.get('complexity', 0) > 1.0:
            score *= 1.1
        
        # Scope relevance
        if chunk.metadata.get('scope'):
            scope_relevance = sum(
                1.0 if any(f_name in s[1] for _, f_name in query_context.get('focus', []))
                else 0.2
                for s in chunk.metadata['scope']
            )
            score *= (1.0 + scope_relevance * 0.3)
        
        # Dependency relevance
        if chunk.metadata.get('dependencies'):
            dep_matches = sum(
                1.0 if dep in query_context['code_elements'].get('imports', [])
                else 0.5 if dep in query_context['code_elements'].get('functions', [])
                else 0.2
                for dep in chunk.metadata['dependencies']
            )
            score *= (1.0 + dep_matches * 0.2)
        
        return score

    def _build_enhanced_context(self, chunks: List[SemanticChunk], query_context: Dict) -> str:
        """Build enhanced context with structured information and relevance data"""
        context_parts = ["## Code Context Summary\n"]
        
        # Group chunks by file
        file_chunks = defaultdict(list)
        file_scores = defaultdict(list)
        
        for chunk in chunks:
            file_path = chunk.metadata.get('file', 'unknown')
            file_chunks[file_path].append(chunk)
            # Get the similarity score from the chunk metadata
            similarity_score = chunk.metadata.get('similarity_score', 0.0)
            file_scores[file_path].append(similarity_score)
        
        # Sort files by average relevance score
        sorted_files = sorted(
            file_scores.keys(),
            key=lambda f: sum(file_scores[f]) / len(file_scores[f]) if file_scores[f] else 0,
            reverse=True
        )
        
        for file_path in sorted_files:
            # Calculate average score for the file
            avg_score = sum(file_scores[file_path]) / len(file_scores[file_path]) if file_scores[file_path] else 0
            # Format as percentage with 2 decimal places
            score_pct = f"{avg_score * 100:.2f}%"
            
            context_parts.append(f"\n### File: `{file_path}`\n**Relevance Score: {score_pct}**")
            
            # Display this to the user in the console with professional formatting
            self.console.print(f"[bold blue]Analyzing File:[/] [bold white]{file_path}[/] [yellow][Relevance: {score_pct}][/]")
            
            # Add file-level imports first
            all_imports = set()
            for chunk in file_chunks[file_path]:
                if chunk.metadata.get('type') == 'import_section':
                    context_parts.append("\n#### Import Statements")
                    context_parts.append("```python")
                    context_parts.extend(chunk.metadata.get('imports', []))
                    context_parts.append("```")
                    all_imports.update(chunk.metadata.get('imports', []))
            
            # Process remaining chunks
            # Sort chunks by similarity score (descending)
            sorted_chunks = sorted(
                [c for c in file_chunks[file_path] if c.metadata.get('type') != 'import_section'],
                key=lambda c: c.metadata.get('similarity_score', 0.0),
                reverse=True
            )
            
            # Add each code chunk with proper headers and structure
            for i, chunk in enumerate(sorted_chunks):
                similarity_score = chunk.metadata.get('similarity_score', 0.0)
                score_pct = f"{similarity_score * 100:.2f}%"
                
                start_line = chunk.metadata.get('start_line', 0)
                end_line = chunk.metadata.get('end_line', 0)
                
                # Create a professional-looking section header
                chunk_type = chunk.metadata.get('type', 'code').replace('_', ' ').title()
                context_parts.append(f"\n#### {chunk_type} (Lines {start_line}-{end_line})")
                context_parts.append(f"**Relevance: {score_pct}**")
                
                # Display chunk details to the console
                self.console.print(f"  [dim]→ {chunk_type} [lines {start_line}-{end_line}] [yellow]Relevance: {score_pct}[/]")
                
                # Add code with line numbers and syntax highlighting
                context_parts.append("```python")
                lines = chunk.content.splitlines()
                for j, line in enumerate(lines, start=start_line):
                    # Add properly formatted line numbers
                    context_parts.append(f"{j:4d}| {line}")
                context_parts.append("```")
                
                # Add any additional context from the chunk
                if 'contexts' in chunk.metadata and chunk.metadata['contexts']:
                    for context_type, context_content in chunk.metadata['contexts'].items():
                        if context_content:
                            context_label = context_type.replace('_', ' ').title()
                            context_parts.append(f"\n**{context_label}:**")
                            context_parts.append(f"```\n{context_content}\n```")
                
                # Limit the number of chunks per file to avoid overwhelming context
                if i >= 4:  # Show max 5 chunks per file
                    remaining = len(sorted_chunks) - i - 1
                    if remaining > 0:
                        context_parts.append(f"\n*({remaining} more code sections omitted for brevity)*")
                    break
        
        # Add query-specific context information
        if query_context:
            context_parts.append("\n### Query Analysis")
            for key, value in query_context.items():
                if key != 'keywords' and value:
                    context_parts.append(f"**{key.replace('_', ' ').title()}:** {value}")
        
        return "\n".join(context_parts)

    def _build_simple_context(self, chunks, query_context: Dict) -> str:
        """Build a simple context from raw document strings or chunks with minimal metadata."""
        context_parts = []
        
        # Group chunks by file
        file_chunks = {}
        file_scores = {}  # Track scores per file
        
        for chunk in chunks:
            # Get file path from chunk metadata or use a default
            file_path = chunk.metadata.get('file', 'unknown_file')
            
            # Get similarity score if available
            similarity_score = chunk.metadata.get('similarity_score', 0.0)
            
            if file_path not in file_chunks:
                file_chunks[file_path] = []
                file_scores[file_path] = []
            
            # Store the chunk content
            content = chunk.content
            
            # Get line numbers if available
            start_line = chunk.metadata.get('start_line', 1)
            end_line = chunk.metadata.get('end_line', start_line + content.count('\n'))
            
            file_chunks[file_path].append((content, start_line, end_line, similarity_score))
            file_scores[file_path].append(similarity_score)
        
        # Sort files by average similarity score (descending)
        sorted_files = sorted(
            file_chunks.keys(),
            key=lambda f: sum(file_scores[f]) / len(file_scores[f]) if file_scores[f] else 0,
            reverse=True
        )
        
        # Process each file's chunks
        for file_path in sorted_files:
            # Calculate average score for the file
            avg_score = sum(file_scores[file_path]) / len(file_scores[file_path]) if file_scores[file_path] else 0
            # Format as percentage with 2 decimal places
            score_pct = f"{avg_score * 100:.2f}%"
            
            context_parts.append(f"\n# File: {file_path} (Relevance: {score_pct})")
            
            # Display this to the user in the console
            self.console.print(f"[bold blue]File: {file_path} [yellow]Relevance: {score_pct}[/]")
            
            # Sort chunks by similarity score (descending)
            sorted_chunks = sorted(file_chunks[file_path], key=lambda c: c[3], reverse=True)
            
            # Add each chunk with line numbers
            for content, start_line, end_line, similarity_score in sorted_chunks:
                # Format score as percentage
                score_pct = f"{similarity_score * 100:.2f}%"
                
                context_parts.append(f"\n## Code (Lines {start_line}-{end_line}, Relevance: {score_pct})")
                context_parts.append("```python")
                
                # Display chunk score in console
                self.console.print(f"  [dim]Chunk lines {start_line}-{end_line} [yellow]Relevance: {score_pct}[/]")
                
                # Add line numbers to the code
                lines = content.splitlines()
                for i, line in enumerate(lines, start=start_line):
                    context_parts.append(f"{i}: {line}")
                
                context_parts.append("```")
        
        return "\n".join(context_parts)

    def _determine_optimal_chunk_count(self, question: str, context: Dict) -> int:
        """Determine optimal number of chunks based on query complexity."""
        base_count = 5  # Start with a smaller base
        
        # Adjust for query type
        type_multipliers = {
            'explain': 2.0,    # Need more context for explanations
            'find': 1.5,       # Need a broader search
            'modify': 1.8,     # Need surrounding context
            'create': 1.5,     # Need examples
            'error': 2.0       # Need more context for debugging
        }
        
        multiplier = type_multipliers.get(context['type'], 1.0)
        
        # Adjust for complexity factors
        complexity_score = (
            len(context['focus']) * 0.5 +  # More focus areas need more chunks
            sum(len(elements) for elements in context['code_elements'].values()) * 0.3 +  # More code elements need more chunks
            (2.0 if any(term in question.lower() for term in 
                       ['architecture', 'structure', 'overall', 'entire', 'all']) else 0)  # System-level questions need more chunks
        )
        
        # Calculate final count
        chunk_count = int(base_count * multiplier * (1 + complexity_score))
        
        # Keep within reasonable bounds
        return min(max(chunk_count, 3), 20)  # Tighter bounds for more focused results

    async def _generate_enhanced_response(self, question: str, context: str,
                                       history_context: str, query_context: Dict) -> str:
        """Generate response with enhanced prompt engineering"""
        # Build enhanced prompt
        prompt_parts = [
            "You are an AI assistant analyzing code. Focus only on the provided context and be direct and accurate.",
            f"Question type: {query_context['type']}",
        ]
        
        # Add focus areas if any
        if query_context['focus']:
            prompt_parts.append(f"Focus areas: {', '.join(f'{t}: {n}' for t, n in query_context['focus'])}")
        
        # Add code elements if present
        code_elements = [f"- {k}: {', '.join(v)}" for k, v in query_context['code_elements'].items() if v]
        if code_elements:
            prompt_parts.append("Code elements mentioned:")
            prompt_parts.extend(code_elements)
        
        # Add the question
        prompt_parts.extend([
            "\nQuestion:",
            question,
        ])
        
        # Check if context is too large and needs to be truncated
        if len(context) > 10000:  # Reduced from 12000 to be more conservative
            console.print("[yellow]Context is very large, truncating to fit token limits[/]")
            # Simple truncation approach - keep the beginning which usually has the most relevant info
            context = context[:10000] + "\n\n[Context truncated due to size]"
        
        # Add the context
        prompt_parts.extend([
            "\nRelevant code context:",
            context
        ])
        
        # Add conversation history if relevant
        if history_context:
            prompt_parts.extend([
                "\nRelevant conversation history:",
                history_context
            ])
        
        # Add final instructions - keeping it simpler
        prompt_parts.extend([
            "\nFormat your response with markdown. Include file paths and line numbers when referencing code."
        ])
        
        enhanced_prompt = "\n".join(filter(None, prompt_parts))
        
        # Generate response with enhanced prompt
        response = await self.ai_client.get_completion(
            enhanced_prompt,
            temperature=self.config.get('temperature', 0.7)
        )
        
        # Format the response for display
        self._format_output(response)
        
        return response
        
    def _enhance_code_blocks(self, text: str) -> str:
        """Enhance code blocks in markdown to look more professional"""
        import re
        
        # Pattern to find code blocks
        code_block_pattern = r"```(\w+)?\n(.*?)```"
        
        def replace_code_block(match):
            lang = match.group(1) or "text"
            code = match.group(2)
            
            # Add line numbers and improve formatting for Python code
            if lang.lower() == "python":
                lines = code.split("\n")
                numbered_lines = []
                
                for i, line in enumerate(lines, 1):
                    # Only add line number if the line has content
                    if line.strip():
                        # Format indentation consistently
                        indentation = len(line) - len(line.lstrip())
                        formatted_line = " " * indentation + line.lstrip()
                        numbered_lines.append(f"{i:3d}| {formatted_line}")
                    else:
                        numbered_lines.append(f"   | ")
                
                enhanced_code = "\n".join(numbered_lines)
                return f"```{lang}\n{enhanced_code}\n```"
            
            return match.group(0)
        
        # Replace code blocks with enhanced versions
        enhanced_text = re.sub(code_block_pattern, replace_code_block, text, flags=re.DOTALL)
        return enhanced_text

    def _update_conversation_history(self, question: str, response: str, query_context: Dict):
        """Update conversation history with context"""
        self.conversation_history.append({
            "question": question,
            "answer": response,
            "context": query_context
        })
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

    def _prepare_filtered_history(self, question: str, query_context: Dict) -> str:
        """Prepare filtered conversation history for context"""
        if 'keywords' in query_context and query_context['keywords']:
            # Filter by keywords if available
            relevant_history = [
                exchange for exchange in self.conversation_history
                if any(term in exchange['question'].lower() for term in query_context['keywords'])
            ]
        else:
            # Otherwise use all history up to a limit
            relevant_history = self.conversation_history[-self.config.get('max_history', 5):]
            
        return self._prepare_conversation_history()

    def _analyze_query(self, question: str) -> Dict:
        """Analyze query to extract semantic information"""
        context = {
            'type': 'general',
            'focus': [],
            'scope': [],
            'keywords': set(),
            'code_elements': {
                'classes': set(),
                'functions': set(),
                'variables': set(),
                'imports': set()
            }
        }
        
        # Identify query type
        type_patterns = {
            'explain': r'explain|how|what|why|describe',
            'find': r'find|where|locate|search',
            'modify': r'change|modify|update|fix',
            'create': r'create|add|implement|make',
            'error': r'error|bug|issue|problem|fail'
        }
        
        for qtype, pattern in type_patterns.items():
            if re.search(pattern, question.lower()):
                context['type'] = qtype
                break
        
        # Extract code elements
        code_patterns = {
            'classes': r'class\s+(\w+)',
            'functions': r'def\s+(\w+)|function\s+(\w+)',
            'variables': r'\b[a-z_][a-z0-9_]*\b',
            'imports': r'import\s+(\w+)|from\s+(\w+)'
        }
        
        for element_type, pattern in code_patterns.items():
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                context['code_elements'][element_type].update(
                    m[0] if isinstance(m, tuple) else m for m in matches
                )
        
        # Extract focus areas
        focus_patterns = [
            (r'in\s+(\w+[\w./]*)', 'file'),
            (r'class\s+(\w+)', 'class'),
            (r'function\s+(\w+)', 'function'),
            (r'method\s+(\w+)', 'method')
        ]
        
        for pattern, focus_type in focus_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                context['focus'].extend((focus_type, m) for m in matches)
        
        # Extract keywords from the question
        # First add significant words (longer than 3 chars)
        for word in re.findall(r'\b\w+\b', question.lower()):
            if len(word) > 3 and word not in ['what', 'does', 'this', 'that', 'with', 'from', 'have', 'about']:
                context['keywords'].add(word)
                
        # Add all code element names as keywords
        for element_type, elements in context['code_elements'].items():
            context['keywords'].update(elements)
            
        # Add focus area names as keywords
        for _, focus_name in context['focus']:
            context['keywords'].add(focus_name.lower())
            
        # If we have no keywords, add some from the question type
        if not context['keywords'] and context['type'] != 'general':
            context['keywords'].add(context['type'])
            
        # Convert keywords set to list for easier serialization
        context['keywords'] = list(context['keywords'])
        
        return context
    
    def _prepare_conversation_history(self) -> str:
        """Format conversation history for context"""
        if not self.conversation_history:
            return ""
            
        history_parts = []
        for i, exchange in enumerate(self.conversation_history):
            question_text = exchange['question'].strip()
            answer_text = exchange['answer'].strip()
            history_parts.append(f"Question {i+1}:\n{question_text}")
            history_parts.append(f"Answer {i+1}:\n{answer_text}")
            
        return "\n\n".join(history_parts)
    
    def _keyword_search(self, query: str) -> List[str]:
        """Search for files by keywords in query"""
        # Extract potential file-related keywords
        query_lower = query.lower()
        keywords = set()
        
        # Extract words that might be filenames or partial filenames
        for word in re.findall(r'\b\w+\b', query_lower):
            if len(word) > 3:  # Only consider words longer than 3 chars
                keywords.add(word)
        
        # Look for potential file extensions
        for ext in ['.py', '.js', '.html', '.css', '.json', '.tsx', '.ts', '.md']:
            if ext in query_lower or ext[1:] in query_lower:  # e.g. '.py' or 'py'
                keywords.add(ext[1:])  # Add without dot
        
        # Search for matching files
        matches = []
        for keyword in keywords:
            for doc_idx, metadata in enumerate(self.metadata):
                filename = metadata['file'].lower()
                if keyword in filename:
                    matches.append(doc_idx)
        
        return list(set(matches))  # Deduplicate
    
    def _should_index_file(self, file_path: Path) -> bool:
        """Determine if file should be indexed"""
        # First, make the path relative to the repository root to avoid .codeai issues
        try:
            rel_path = file_path.relative_to(self.path)
            parts = rel_path.parts
        except ValueError:
            # If we can't get a relative path, use the original parts
            parts = file_path.parts
        
        file_extension = file_path.suffix.lower()
        
        # Skip hidden directories (but not the .codeai directory itself)
        if any(part.startswith('.') for part in parts) and not (len(parts) == 1 and parts[0] == '.env'):
            console.print(f"[dim]Skipping {rel_path}: hidden directory or file[/]")
            return False
        
        IGNORE_DIRS = {'node_modules', 'venv', 'env', 'build', 'dist', '__pycache__'}
        if any(part in IGNORE_DIRS for part in parts):
            console.print(f"[dim]Skipping {rel_path}: in ignored directory[/]")
            return False
        
        if file_extension == '':
            console.print(f"[dim]Skipping {rel_path}: no file extension[/]")
            return False
            
        # Include common code file extensions, especially frontend-related ones
        CODE_EXTENSIONS = {
            '.py', '.js', '.ts', '.jsx', '.tsx',         # Python, JavaScript, TypeScript
            '.java', '.cpp', '.c', '.h', '.hpp', '.cs',   # Java, C++, C#
            '.go', '.rs',                                 # Go, Rust
            '.html', '.css', '.scss', '.sass',            # Web files
            '.vue', '.svelte',                            # Vue, Svelte
            '.json', '.xml', '.yaml', '.yml',             # Data files
            '.md', '.markdown',                           # Documentation
            '.cjs', '.mjs', '.ejs',                       # Other JS variants
            '.mts', '.cts',                               # TypeScript variants
            '.gitignore', '.env', '.eslintrc'             # Config files
        }
        
        is_indexable = file_extension in CODE_EXTENSIONS
        
        # Debug output for every file
        if is_indexable:
            console.print(f"[green]Including file: {rel_path} with extension {file_extension}[/]")
        else:
            console.print(f"[yellow]Skipping file: {rel_path} with extension '{file_extension}'[/]")
        
        return is_indexable
    
    def _chunk_file(self, file_path: Path) -> List[SemanticChunk]:
        """Split file into semantic chunks with enhanced context tracking."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get file type and basic structure
            file_type = file_path.suffix.lstrip('.')
            structure = extract_code_structure(content, file_type)
            
            chunks = []
            lines = content.splitlines(keepends=True)
            
            # Enhanced context tracking
            current_class = None
            current_function = None
            current_scope = []
            import_section = []
            
            i = 0
            while i < len(lines):
                # Determine chunk type and boundaries
                chunk_start = i
                chunk_type = 'unknown'
                
                # Look ahead for semantic boundaries
                lookahead = ''.join(lines[i:i+5])  # Look at next 5 lines
                
                if re.match(r'^\s*(class|interface)\s+\w+', lookahead):
                    chunk_type = 'class_definition'
                    current_class = re.match(r'^\s*(class|interface)\s+(\w+)', lines[i]).group(2)
                    current_scope.append(('class', current_class))
                    
                elif re.match(r'^\s*(def|function|async def)\s+\w+', lookahead):
                    chunk_type = 'function_definition'
                    current_function = re.match(r'^\s*(def|function|async def)\s+(\w+)', lines[i]).group(2)
                    current_scope.append(('function', current_function))
                    
                elif re.match(r'^\s*(import|from)\s+\w+', lookahead):
                    chunk_type = 'import_section'
                    import_section.append(lines[i])
                    
                # Find chunk end based on type
                chunk_end = self._find_semantic_boundary(lines, i, chunk_type)
                
                # Create chunk content
                chunk_lines = lines[chunk_start:chunk_end]
                chunk_content = ''.join(chunk_lines)
                
                # Calculate complexity
                complexity = self._calculate_complexity(chunk_content)
                
                # Create chunk metadata
                metadata = {
                    'file': str(file_path.relative_to(self.path)),
                    'start_line': chunk_start + 1,
                    'end_line': chunk_end,
                    'type': chunk_type,
                    'scope': current_scope.copy(),
                    'complexity': complexity,
                    'has_docstring': bool(re.search(r'""".*?"""', chunk_content, re.DOTALL)),
                    'has_comments': bool(re.search(r'#.*$', chunk_content, re.MULTILINE)),
                    'language': file_type,
                    'imports': structure.imports,
                    'dependencies': self._extract_dependencies(chunk_content)
                }
                
                # Create semantic chunk
                chunk = SemanticChunk(chunk_content, metadata)
                
                # Add relevant context
                if current_class:
                    chunk.add_context('class', f"Class: {current_class}")
                if current_function:
                    chunk.add_context('function', f"Function: {current_function}")
                if import_section:
                    chunk.add_context('imports', ''.join(import_section))
                
                # Calculate importance score
                chunk.calculate_importance()
                
                chunks.append(chunk)
                
                # Update position
                i = chunk_end
                
                # Update scope
                if chunk_type == 'class_definition' and current_scope[-1][0] == 'class':
                    current_scope.pop()
                    current_class = current_scope[-1][1] if current_scope and current_scope[-1][0] == 'class' else None
                elif chunk_type == 'function_definition' and current_scope[-1][0] == 'function':
                    current_scope.pop()
                    current_function = current_scope[-1][1] if current_scope and current_scope[-1][0] == 'function' else None
            
            return chunks
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not process {file_path}: {str(e)}[/]")
            return []

    def _find_semantic_boundary(self, lines: List[str], start: int, chunk_type: str) -> int:
        """Find the semantic boundary for a chunk based on its type."""
        if chunk_type == 'class_definition':
            # Find the end of the class (accounting for nested classes)
            depth = 0
            for i in range(start, len(lines)):
                if re.match(r'^\s*class\s+', lines[i]):
                    depth += 1
                elif re.match(r'^\S', lines[i]) and depth > 0:
                    depth -= 1
                    if depth == 0:
                        return i
            return len(lines)
        
        elif chunk_type == 'function_definition':
            # Find the end of the function (accounting for nested functions)
            depth = 0
            for i in range(start, len(lines)):
                if re.match(r'^\s*(def|async def)\s+', lines[i]):
                    depth += 1
                elif re.match(r'^\S', lines[i]) and depth > 0:
                    depth -= 1
                    if depth == 0:
                        return i
            return len(lines)
        
        elif chunk_type == 'import_section':
            # Find the end of consecutive import statements
            for i in range(start, len(lines)):
                if not re.match(r'^\s*(import|from)\s+\w+', lines[i]):
                    return i
            return len(lines)
        
        else:
            # Default chunking based on blank lines and indentation
            chunk_size = self.config.get('chunk_size', 20)
            end = min(start + chunk_size, len(lines))
            
            # Look for a good boundary
            for i in range(end-1, start, -1):
                # Prefer blank lines
                if not lines[i].strip():
                    return i + 1
                # Or lines with same indentation as start
                if len(lines[i]) - len(lines[i].lstrip()) == len(lines[start]) - len(lines[start].lstrip()):
                    return i + 1
            
            return end

    def _calculate_complexity(self, content: str) -> float:
        """Calculate code complexity score."""
        score = 0.0
        
        # Control flow complexity
        control_flows = len(re.findall(r'\b(if|else|elif|for|while|try|except)\b', content))
        score += control_flows * 0.2
        
        # Nesting complexity
        max_indent = 0
        for line in content.splitlines():
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent)
        score += (max_indent / 4) * 0.1
        
        # Length complexity
        lines = len(content.splitlines())
        score += min(lines / 10, 2.0)
        
        return score

    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract code dependencies from content."""
        dependencies = set()
        
        # Extract imported names
        import_matches = re.finditer(r'(?:from\s+(\w+)\s+import\s+(\w+))|(?:import\s+(\w+))', content)
        for match in import_matches:
            if match.group(1) and match.group(2):  # from ... import ...
                dependencies.add(f"{match.group(1)}.{match.group(2)}")
            elif match.group(3):  # import ...
                dependencies.add(match.group(3))
        
        # Extract function calls
        func_calls = re.findall(r'(\w+)\s*\(', content)
        dependencies.update(func_calls)
        
        # Extract class usage
        class_usage = re.findall(r'(\w+)\s*\.\s*\w+', content)
        dependencies.update(class_usage)
        
        return list(dependencies)

    def save_state(self):
        """Save the current state of the analyzer, including the FAISS index, documents, and metadata."""
        try:
            # Save FAISS index
            faiss_index_path = self.project_dir / "faiss_index.bin"
            faiss.write_index(self.faiss_index, str(faiss_index_path))
            console.print(f"[green]FAISS index saved to {faiss_index_path}[/]")

            # Save documents and metadata
            documents_path = self.project_dir / "documents.pkl"
            metadata_path = self.project_dir / "metadata.pkl"
            with open(documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            console.print(f"[green]Documents and metadata saved to {documents_path} and {metadata_path}[/]")
            
            # Save conversation history
            history_path = self.project_dir / "conversation_history.pkl"
            with open(history_path, 'wb') as f:
                pickle.dump(self.conversation_history, f)
            console.print(f"[green]Conversation history saved with {len(self.conversation_history)} exchanges[/]")
            
            # Save codebase summary if it exists
            if hasattr(self, 'codebase_summary'):
                summary_path = self.project_dir / "codebase_summary.md"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(self.codebase_summary)
                console.print(f"[green]Codebase summary saved to {summary_path}[/]")
        except Exception as e:
            console.print(f"[bold red]Error saving state: {str(e)}[/]")
            traceback.print_exc()
            
    def load_state(self):
        """Load the saved state of the analyzer, including the FAISS index, documents, and metadata."""
        try:
            # Load FAISS index
            faiss_index_path = self.project_dir / "faiss_index.bin"
            if not faiss_index_path.exists():
                raise FileNotFoundError(f"FAISS index file not found at {faiss_index_path}")
            
            self.faiss_index = faiss.read_index(str(faiss_index_path))
            console.print(f"[green]FAISS index loaded from {faiss_index_path}[/]")

            # Load documents and metadata
            documents_path = self.project_dir / "documents.pkl"
            metadata_path = self.project_dir / "metadata.pkl"
            
            if not documents_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(f"Documents or metadata files not found")
            
            with open(documents_path, 'rb') as f:
                self.documents = pickle.load(f)
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
                
            console.print(f"[green]Documents and metadata loaded from {documents_path} and {metadata_path}[/]")
            
            # Load conversation history if it exists
            history_path = self.project_dir / "conversation_history.pkl"
            if history_path.exists():
                with open(history_path, 'rb') as f:
                    self.conversation_history = pickle.load(f)
                console.print(f"[green]Conversation history loaded with {len(self.conversation_history)} exchanges[/]")
            else:
                self.conversation_history = []
            
            # Load codebase summary if it exists
            summary_path = self.project_dir / "codebase_summary.md"
            if summary_path.exists():
                with open(summary_path, 'r', encoding='utf-8') as f:
                    self.codebase_summary = f.read()
                console.print(f"[green]Codebase summary loaded from {summary_path}[/]")
            else:
                self.codebase_summary = "# Codebase Summary\n\nNo detailed summary available."
                
        except FileNotFoundError as e:
            console.print(f"[bold red]Error loading state: {str(e)}[/]")
            console.print("[yellow]Have you indexed the codebase yet? Try running 'python cli.py init' first.[/]")
            raise
        except Exception as e:
            console.print(f"[bold red]Error loading state: {str(e)}[/]")
            traceback.print_exc()
            raise

    async def _generate_codebase_summary(self, indexable_files):
        """Generate a high-level summary of the codebase structure."""
        try:
            # Collect key files and their purposes
            key_files = []
            file_count_by_type = {}
            directory_structure = {}
            
            # Analyze file types and directory structure
            for file_path in indexable_files:
                rel_path = str(file_path.relative_to(self.path))
                ext = file_path.suffix.lower()
                file_count_by_type[ext] = file_count_by_type.get(ext, 0) + 1
                
                # Track directory structure
                parts = rel_path.split(os.sep)
                current_level = directory_structure
                for i, part in enumerate(parts[:-1]):  # Process directories
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]
                
                # Identify potential key files
                filename = file_path.name.lower()
                if any(key in filename for key in ['main', 'app', 'index', 'server', 'config', 'core']):
                    # Read first few lines to get a sense of the file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            first_lines = ''.join(f.readline() for _ in range(10))
                        key_files.append({
                            'path': rel_path,
                            'preview': first_lines[:200] + '...' if len(first_lines) > 200 else first_lines
                        })
                    except Exception:
                        key_files.append({'path': rel_path, 'preview': 'Could not read file'})
            
            # Generate a textual summary
            summary_parts = [
                "# Codebase Summary",
                f"\n## Overview",
                f"Total files: {len(indexable_files)}",
                f"File types: {', '.join(f'{ext} ({count})' for ext, count in sorted(file_count_by_type.items(), key=lambda x: x[1], reverse=True))}",
                
                f"\n## Key Files",
            ]
            
            for file_info in key_files[:10]:  # Limit to 10 key files
                summary_parts.append(f"\n### {file_info['path']}")
                summary_parts.append(f"```\n{file_info['preview']}\n```")
            
            # Add directory structure
            summary_parts.append(f"\n## Directory Structure")
            
            def format_directory(dir_dict, prefix=''):
                lines = []
                for name, contents in dir_dict.items():
                    lines.append(f"{prefix}📁 {name}/")
                    if contents:
                        lines.extend(format_directory(contents, prefix + '  '))
                return lines
            
            summary_parts.extend(format_directory(directory_structure))
            
            # Store the summary
            self.codebase_summary = '\n'.join(summary_parts)
            console.print(f"[green]Generated codebase summary ({len(summary_parts)} sections)[/]")
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate complete codebase summary: {str(e)}[/]")
            self.codebase_summary = f"# Codebase Summary\n\nTotal files: {len(indexable_files)}"