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
from parallel_processor import ParallelProcessor

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
        self.parallel_processor = ParallelProcessor()  # Add parallel processor
        
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
        """Index the codebase using parallel processing."""
        try:
            with Progress() as progress:
                # Find all files
                task_find = progress.add_task("[blue]Finding files...", total=1)
                console.print("[blue]Starting indexing process...[/]")
                
                all_files = []
                for root, _, files in os.walk(self.path):
                    for file in files:
                        all_files.append(Path(root) / file)
                        
                progress.update(task_find, advance=1)
                console.print(f"Found {len(all_files)} files in total")
                
                # Show sample of files found
                if all_files:
                    sample_files = all_files[:5]
                    console.print("Sample files:", ", ".join(str(f) for f in sample_files), "...")
                
                # Filter indexable files
                task_filter = progress.add_task("[blue]Filtering indexable files...", total=1)
                
                indexable_files = []
                file_extensions = {}
                
                for file in all_files:
                    file_path = str(file)
                    
                    # Skip hidden files and directories, but allow .codeai/repos
                    parts = Path(file_path).parts
                    if any(part.startswith('.') and part != '.codeai' and not (part == '.git' and 'repos' in parts) for part in parts):
                        console.print(f"Skipping {file_path}: hidden directory or file")
                        continue
                    
                    # Skip .git directory contents
                    if '.git' in parts and 'repos' not in parts:
                        console.print(f"Skipping {file_path}: git directory")
                        continue
                        
                    # Get file extension
                    ext = file.suffix.lower()
                    
                    # Track extension counts
                    if ext:
                        file_extensions[ext] = file_extensions.get(ext, 0) + 1
                    
                    # Include files with supported extensions
                    if ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.cs', '.go', '.rs', '.md', '.txt']:
                        console.print(f"Including file: {file_path} with extension {ext}")
                        indexable_files.append(file)
                
                progress.update(task_filter, advance=1)
                
                # Print extension breakdown
                console.print(f"\nFound {len(indexable_files)} indexable files")
                if file_extensions:
                    console.print("Extension breakdown:")
                    for ext, count in sorted(file_extensions.items(), key=lambda x: x[1], reverse=True):
                        console.print(f"  {ext}: {count} files")
                
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
                
                # Process files in parallel and generate embeddings
                all_embeddings, all_documents, all_metadata = await self.parallel_processor.process_files_parallel(
                    files=indexable_files,
                    chunk_size=self.config.get('chunk_size', 20),
                    embedding_client=self.ai_client,
                    batch_size=self.config.get('parallel', {}).get('batch_size', 10),
                    progress=progress
                )
                
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
                    
                    console.print(f"[bold green]Indexing complete! Processed {len(all_documents)} chunks from {len(indexable_files)} files.[/]")
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
        """Format text using a simple, conversational style"""
        try:
            # Simply print the text without any fancy formatting
            self.console.print(text)
        except Exception as e:
            self.console.print(f"[bold red]Error formatting output:[/] {str(e)}")
            # Fallback to simple printing
            self.console.print(text)

    async def query(self, question: str, chunk_count: int = None) -> str:
        """Query the codebase with enhanced semantic understanding."""
        try:
            # Analyze query intent and context
            query_context = self._analyze_query(question)
            
            # Check for existing file content from follow-up questions
            if 'file_content' in query_context and query_context.get('is_file_query', False):
                target_file_name = query_context.get('target_file')
                file_content = query_context.get('file_content')
                
                self.console.print(f"[bold green]Using previously found file content for follow-up: {target_file_name}[/]")
                direct_file_context = self._create_direct_file_context(target_file_name, file_content)
                
                # Generate response with direct file context
                history_context = ""
                if self.conversation_history:
                    history_context = self._prepare_filtered_history(question, query_context)
                
                response = await self._generate_direct_file_response(
                    question,
                    direct_file_context,
                    history_context,
                    query_context
                )
                
                # Update conversation history
                self._update_conversation_history(question, response, query_context)
                
                return response
            
            # For file-specific queries, directly extract the file content if it exists
            if query_context.get('is_file_query', False) and query_context.get('target_file'):
                target_file_name = query_context.get('target_file')
                file_content = self._find_and_extract_file_content(target_file_name)
                
                if file_content:
                    # If file was found, create a special direct file context
                    self.console.print(f"[bold green]File found directly: {target_file_name}[/]")
                    direct_file_context = self._create_direct_file_context(target_file_name, file_content)
                    
                    # Save the file content in the query context for follow-up questions
                    query_context['file_content'] = file_content
                    
                    # Generate response with direct file context
                    history_context = ""
                    if self.conversation_history:
                        history_context = self._prepare_filtered_history(question, query_context)
                    
                    response = await self._generate_direct_file_response(
                        question,
                        direct_file_context,
                        history_context,
                        query_context
                    )
                    
                    # Update conversation history
                    self._update_conversation_history(question, response, query_context)
                    
                    return response
                else:
                    self.console.print(f"[yellow]File not found directly: {target_file_name}, continuing with semantic search[/]")
                    # Continue with regular semantic search if file not found directly
            
            # Continue with regular semantic search
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

    def _find_and_extract_file_content(self, target_file_name: str) -> Optional[str]:
        """Find a file by name in the repository and extract its contents."""
        try:
            # Search for files that match the target name (case-insensitive)
            matched_files = []
            for root, _, files in os.walk(self.path):
                for file in files:
                    if file.lower() == target_file_name.lower():
                        matched_files.append(Path(root) / file)
            
            if not matched_files:
                # Try partial matching if no exact match
                for root, _, files in os.walk(self.path):
                    for file in files:
                        if target_file_name.lower() in file.lower():
                            matched_files.append(Path(root) / file)
            
            if matched_files:
                # Use the first match (prioritize exact matches if any)
                file_path = matched_files[0]
                
                # Log the file being used
                console.print(f"[blue]Reading file: {file_path}[/]")
                
                # Read and return the content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return content
            
            return None
        except Exception as e:
            console.print(f"[yellow]Error reading file: {str(e)}[/]")
            return None

    def _create_direct_file_context(self, file_name: str, content: str) -> str:
        """Create a context with the complete file content for direct file queries."""
        lines = content.splitlines()
        
        # Create a nicely formatted context with the full file content
        context_parts = [
            f"## File: {file_name}",
            "\nThis is the complete content of the requested file:\n"
        ]
        
        # Add the file content with line numbers
        context_parts.append("```python")
        for i, line in enumerate(lines, 1):
            context_parts.append(f"{i:4d}| {line}")
        context_parts.append("```\n")
        
        # Try to add a brief analysis of the file structure
        try:
            file_type = Path(file_name).suffix.lstrip('.')
            structure = extract_code_structure(content, file_type)
            
            if structure.imports:
                context_parts.append("### Imports")
                for imp in structure.imports:
                    context_parts.append(f"- `{imp}`")
                context_parts.append("")
            
            if structure.functions:
                context_parts.append("### Functions")
                for func in structure.functions:
                    context_parts.append(f"- `{func}()`")
                context_parts.append("")
            
            if structure.classes:
                context_parts.append("### Classes")
                for class_name, details in structure.classes.items():
                    context_parts.append(f"- `{class_name}`")
                    if details['methods']:
                        context_parts.append("  Methods:")
                        for method in details['methods']:
                            context_parts.append(f"  - `{method}()`")
                    if details['attributes']:
                        context_parts.append("  Attributes:")
                        for attr in details['attributes']:
                            context_parts.append(f"  - `{attr}`")
                context_parts.append("")
        except Exception as e:
            console.print(f"[yellow]Error analyzing file structure: {str(e)}[/]")
        
        return "\n".join(context_parts)

    async def _generate_direct_file_response(self, question: str, file_context: str,
                                          history_context: str, query_context: Dict) -> str:
        """Generate response for direct file queries with complete file content."""
        # Build a specialized prompt for file-specific questions
        prompt_parts = [
            "You are a friendly and helpful AI assistant analyzing code. Explain things in a natural, conversational way.",
            "You're looking at a specific file that the user asked about.",
            f"The file is: {query_context.get('target_file')}",
            "Please be thorough but explain things like you're having a casual conversation with a fellow developer.",
            "In your response, try to:"
        ]

        # Add specialized instructions based on file type
        file_extension = Path(query_context.get('target_file', '')).suffix.lower()
        
        # Add final instructions
        prompt_parts.extend([
            "\nWhen explaining this file:",
            "1. Start with a friendly overview of what the file does",
            "2. Point out the interesting parts of the code",
            "3. Explain how things work in a clear, conversational way",
            "4. Mention any important patterns or things to watch out for",
            "5. Use examples if they help explain things better",
            "6. Keep it thorough but natural - like you're explaining to a colleague"
        ])
        
        # Add the question
        prompt_parts.extend([
            "\nUser's question:",
            question,
            "\nNOTE: Even for simple questions, you must provide a detailed and thorough analysis. Simple questions deserve comprehensive answers."
        ])
        
        # Add the file context (which contains the complete file)
        prompt_parts.extend([
            "\nFile content:",
            file_context
        ])
        
        # Add conversation history if relevant
        if history_context:
            prompt_parts.extend([
                "\nRelevant conversation history:",
                history_context
            ])
        
        # Add final instructions
        prompt_parts.extend([
            "\nCRITICAL INSTRUCTIONS FOR RESPONSE:",
            "1. Your response MUST be at least 800 words in length to be sufficiently detailed",
            "2. Start with a clear, extensive summary of what this specific file does",
            "3. Explain ALL the actual code snippets from the file, showing the most important parts",
            "4. Provide extremely detailed explanations of how the code in this file works",
            "5. Analyze the implementation details, algorithmic choices, and code structure",
            "6. Reference specific line numbers from the file when explaining code",
            "7. Format your response with proper markdown for readability",
            "8. Include code examples with explanations for complex parts",
            "9. Make sure your response directly answers the user's question about this file",
            "10. If you find yourself providing a short answer, EXPAND it with more analysis, examples, and details"
        ])
        
        enhanced_prompt = "\n".join(filter(None, prompt_parts))
        
        # Generate response with enhanced prompt
        response = await self.ai_client.get_completion(
            enhanced_prompt,
            temperature=self.config.get('temperature', 0.7),
            max_tokens=4000  # Ensure we have enough tokens for a detailed response
        )
        
        # Check if response is too short and try again with stronger instructions if needed
        if len(response.split()) < 200:  # If response is less than ~200 words
            console.print("[yellow]Response too short, requesting a more detailed response...[/]")
            
            # Add even stronger instructions for detailed response
            prompt_parts.append("\nWARNING: Your previous answer was too brief. You MUST provide a MUCH more detailed analysis.")
            prompt_parts.append("Make sure to cover EVERY function, class, and code block in the file with thorough explanation.")
            prompt_parts.append("Your response should be at least 1000 words to be considered adequate.")
            
            enhanced_prompt = "\n".join(filter(None, prompt_parts))
            
            # Try again with stronger instructions
            response = await self.ai_client.get_completion(
                enhanced_prompt,
                temperature=self.config.get('temperature', 0.8),  # Slightly higher temperature
                max_tokens=4000
            )
        
        # Format the response for display
        self._format_output(response, f"Analysis of {query_context.get('target_file')}")
        
        return response

    def _calculate_chunk_relevance(self, chunk: SemanticChunk, query_context: Dict, embedding_distance: float) -> float:
        """Calculate chunk relevance score using multiple factors."""
        # Start with embedding similarity (convert distance to similarity)
        score = 1.0 / (1.0 + embedding_distance)
        
        # Boost by chunk importance
        score *= (1.0 + chunk.importance_score * 0.5)
        
        # Give substantial boost to exact file matches when asking about specific files
        if query_context.get('is_file_query', False) and query_context.get('target_file'):
            chunk_file = Path(chunk.metadata.get('file', '')).name
            target_file = query_context.get('target_file')
            
            # Direct file match gets a very large boost
            if chunk_file.lower() == target_file.lower():
                # Apply 5x boost for direct file match
                score *= 5.0
                console.print(f"[green]Boosting relevance for file match: {chunk_file}[/]")
        
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
        
        # Always use more chunks for file-specific queries
        if context.get('is_file_query', False):
            base_count = 10  # Higher base count for file queries
            self.console.print(f"[blue]File-specific query detected for {context.get('target_file')}. Using increased chunk count.[/]")
        
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
        
        # Different bounds for file-specific queries
        if context.get('is_file_query', False):
            return min(max(chunk_count, 10), 30)  # Higher minimum and maximum for file queries
        
        # Keep within reasonable bounds for general queries
        return min(max(chunk_count, 3), 20)  # Tighter bounds for more focused results

    async def _generate_enhanced_response(self, question: str, context: str,
                                       history_context: str, query_context: Dict) -> str:
        """Generate response with enhanced prompt engineering"""
        # Build enhanced prompt
        prompt_parts = [
            "You are a friendly and helpful AI assistant analyzing code. Your responses should be conversational and easy to understand.",
            "IMPORTANT: Explain things in a natural, chat-like way while still being thorough and detailed.",
            "When discussing code, explain it in a way that feels like a conversation between developers.",
            f"Question type: {query_context['type']}",
            "Remember to be thorough but avoid overly formal or academic language."
        ]
        
        # Add the question and context
        prompt_parts.extend([
            "\nQuestion:",
            question,
            "\nPlease explain in a natural, conversational way. Include:",
            "1. Clear explanations of the relevant code",
            "2. Examples where helpful",
            "3. Important details about how things work",
            "4. Any related components or dependencies worth mentioning"
        ])
        
        # Check if context is too large and needs to be truncated
        if len(context) > 10000:  # Reduced from 12000 to be more conservative
            console.print("[yellow]Context is very large, truncating to fit token limits[/]")
            
            # Intelligent truncation strategy
            if query_context.get('is_file_query', False) and query_context.get('target_file'):
                target_file = query_context.get('target_file')
                sections = context.split("### File: ")
                
                # Start with the header
                if sections[0].startswith("## Code Context Summary"):
                    truncated_context = [sections[0]]
                else:
                    truncated_context = ["## Code Context Summary\n"]
                
                # Separate target file sections from others
                target_sections = []
                other_sections = []
                
                for section in sections[1:]:
                    if target_file.lower() in section.lower():
                        target_sections.append(section)
                    else:
                        other_sections.append(section)
                
                # Add target file sections first
                for section in target_sections:
                    truncated_context.append("### File: " + section)
                
                # Add other sections if there's space
                remaining_length = 10000 - len("\n".join(truncated_context))
                for section in other_sections:
                    section_text = "### File: " + section
                    if len(section_text) < remaining_length:
                        truncated_context.append(section_text)
                        remaining_length -= len(section_text)
                    else:
                        break
                
                # Combine sections
                context = "\n".join(truncated_context)
                
                # Final check if still too long
                if len(context) > 10000:
                    context = context[:10000] + "\n\n[Context truncated due to size]"
            else:
                # General truncation approach
                sections = context.split("### File: ")
                
                # Keep the header
                if sections[0].startswith("## Code Context Summary"):
                    truncated_context = [sections[0]]
                else:
                    truncated_context = ["## Code Context Summary\n"]
                
                # Add sections until limit reached
                remaining_length = 10000 - len(truncated_context[0])
                for section in sections[1:]:
                    section_text = "### File: " + section
                    if len(section_text) < remaining_length:
                        truncated_context.append(section_text)
                        remaining_length -= len(section_text)
                    else:
                        break
                
                # Combine sections
                context = "\n".join(truncated_context)
                
                # Final check if still too long
                if len(context) > 10000:
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
        
        # Add final instructions for comprehensive response
        prompt_parts.extend([
            "\nCRITICAL INSTRUCTIONS FOR RESPONSE:",
            "1. Your response MUST be at least 600 words in length to be sufficiently detailed",
            "2. Start with a clear, extensive overview of the relevant code",
            "3. Include and thoroughly explain actual code snippets from the files",
            "4. Provide extremely detailed explanations of how the code works",
            "5. Analyze implementation details, algorithmic choices, and code structure",
            "6. Reference specific file paths and line numbers",
            "7. Highlight any important patterns, edge cases or considerations",
            "8. Format your response with markdown for readability",
            "9. MAKE SURE TO INCLUDE ALL RELEVANT CODE SNIPPETS FROM THE CONTEXT",
            "10. If you find yourself providing a short answer, EXPAND it with more analysis, examples, and details"
        ])
        
        enhanced_prompt = "\n".join(filter(None, prompt_parts))
        
        # Generate response with enhanced prompt
        response = await self.ai_client.get_completion(
            enhanced_prompt,
            temperature=self.config.get('temperature', 0.7),
            max_tokens=4000  # Ensure we have enough tokens for a detailed response
        )
        
        # Check if response is too short and try again with stronger instructions if needed
        if len(response.split()) < 200:  # If response is less than ~200 words
            console.print("[yellow]Response too short, requesting a more detailed response...[/]")
            
            # Add even stronger instructions for detailed response
            prompt_parts.append("\nWARNING: Your previous answer was too brief. You MUST provide a MUCH more detailed analysis.")
            prompt_parts.append("Make sure to cover ALL relevant code components with thorough explanation.")
            prompt_parts.append("Your response should be at least 800 words to be considered adequate.")
            
            enhanced_prompt = "\n".join(filter(None, prompt_parts))
            
            # Try again with stronger instructions
            response = await self.ai_client.get_completion(
                enhanced_prompt,
                temperature=self.config.get('temperature', 0.8),  # Slightly higher temperature
                max_tokens=4000
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
        # For file-specific follow-up questions, try to find previous exchanges about the same file
        if query_context.get('is_file_query', False) and query_context.get('target_file'):
            # Current file query
            target_file = query_context.get('target_file').lower()
            file_specific_history = []
            
            # Look for previous exchanges about this file
            for exchange in self.conversation_history:
                exchange_context = exchange.get('context', {})
                if exchange_context.get('is_file_query') and exchange_context.get('target_file'):
                    if exchange_context.get('target_file').lower() == target_file:
                        file_specific_history.append(exchange)
            
            if file_specific_history:
                self.console.print(f"[blue]Found {len(file_specific_history)} previous exchanges about {target_file}[/]")
                return self._format_conversation_history(file_specific_history)
        
        # For vague follow-up questions like "explain this", "tell me more", etc.
        follow_up_indicators = ['this', 'that', 'these', 'those', 'it', 'them', 'more', 'explain']
        is_follow_up = any(word in question.lower().split() for word in follow_up_indicators) and len(question.split()) < 6
        
        if is_follow_up and self.conversation_history:
            # Use the most recent exchange as context
            self.console.print("[blue]Detected follow-up question, using most recent exchange for context[/]")
            last_exchange = self.conversation_history[-1]
            follow_up_context = query_context.copy()
            
            # Copy relevant context from the previous question
            prev_context = last_exchange.get('context', {})
            if prev_context.get('is_file_query') and prev_context.get('target_file'):
                follow_up_context['is_file_query'] = True
                follow_up_context['target_file'] = prev_context.get('target_file')
                query_context.update(follow_up_context)  # Update the current query context
                
                # Use this opportunity to try direct file access again for the follow-up
                target_file_name = prev_context.get('target_file')
                file_content = self._find_and_extract_file_content(target_file_name)
                if file_content:
                    self.console.print(f"[bold green]Found file for follow-up question: {target_file_name}[/]")
                    query_context['file_content'] = file_content
            
            return self._format_conversation_history([last_exchange])

        if 'keywords' in query_context and query_context['keywords']:
            # Filter by keywords if available
            relevant_history = [
                exchange for exchange in self.conversation_history
                if any(term in exchange['question'].lower() for term in query_context['keywords'])
            ]
        else:
            # Otherwise use all history up to a limit
            relevant_history = self.conversation_history[-self.config.get('max_history', 5):]
            
        return self._format_conversation_history(relevant_history)

    def _format_conversation_history(self, exchanges: List[Dict]) -> str:
        """Format a list of conversation exchanges for context"""
        if not exchanges:
            return ""
            
        history_parts = []
        for i, exchange in enumerate(exchanges):
            question_text = exchange['question'].strip()
            answer_text = exchange['answer'].strip()
            history_parts.append(f"Question {i+1}:\n{question_text}")
            history_parts.append(f"Answer {i+1}:\n{answer_text}")
        
        return "\n\n".join(history_parts)

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
            },
            'is_file_query': False,  # Flag for file-specific queries
            'target_file': None      # Target file if specified
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
        
        # Check for file-specific queries (with file extensions)
        file_patterns = [
            r'(?:the|a|an)?\s*([a-zA-Z0-9_-]+\.[a-zA-Z0-9]+)\s*(?:file|script|module)?',  # matches example.py
            r'(?:file|script|module)\s+(?:named|called)?\s*([a-zA-Z0-9_-]+\.[a-zA-Z0-9]+)', # matches file named example.py
            r'([a-zA-Z0-9_-]+\.[a-zA-Z0-9]+)\s*(?:file|script|module|code)'  # matches example.py file
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                context['is_file_query'] = True
                context['target_file'] = matches[0]
                # Add file name to focus
                context['focus'].append(('file', matches[0]))
                # Add file name to keywords with high priority
                context['keywords'].add(matches[0])
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
        
        # Skip hidden directories (but allow .codeai/repos and .git in repos)
        if any(part.startswith('.') and part != '.codeai' and not (part == '.git' and 'repos' in parts) for part in parts):
            console.print(f"[dim]Skipping {rel_path}: hidden directory or file[/]")
            return False
            
        # Skip .git directory contents unless in repos
        if '.git' in parts and 'repos' not in parts:
            console.print(f"[dim]Skipping {rel_path}: git directory[/]")
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

    async def refresh_index(self):
        """Refresh the codebase index by only indexing new or modified files."""
        try:
            console.print("[blue]Starting incremental refresh of the codebase index...[/]")
            
            # Make sure we have an existing index to update
            if not hasattr(self, 'faiss_index') or self.faiss_index is None:
                console.print("[yellow]No existing index found. Running full indexing instead...[/]")
                return await self.index()
                
            # Get existing file information from metadata
            indexed_files = {}
            for meta in self.metadata:
                if 'file' in meta and 'last_modified' in meta:
                    indexed_files[meta['file']] = meta['last_modified']
            
            console.print(f"[blue]Found {len(indexed_files)} previously indexed files[/]")
            
            # Find all indexable files in the current codebase
            all_files = []
            for root, _, files in os.walk(self.path):
                for file in files:
                    file_path = Path(root) / file
                    if self._should_index_file(file_path):
                        all_files.append(file_path)
            
            console.print(f"[blue]Found {len(all_files)} indexable files in the current codebase[/]")
            
            # Check which files are new or modified
            new_or_modified_files = []
            file_extensions = {}
            
            for file_path in all_files:
                try:
                    rel_path = str(file_path.relative_to(self.path))
                    last_modified = file_path.stat().st_mtime
                    ext = file_path.suffix.lower()
                    
                    # Track file extensions
                    if ext:
                        file_extensions[ext] = file_extensions.get(ext, 0) + 1
                    
                    # Check if file is new or modified
                    if rel_path not in indexed_files or last_modified > indexed_files[rel_path]:
                        new_or_modified_files.append(file_path)
                        if rel_path in indexed_files:
                            console.print(f"[yellow]Modified file: {rel_path}[/]")
                        else:
                            console.print(f"[green]New file: {rel_path}[/]")
                except Exception as e:
                    console.print(f"[yellow]Error processing file {file_path}: {str(e)}[/]")
            
            # Check for deleted files
            current_files = {str(file_path.relative_to(self.path)) for file_path in all_files}
            deleted_files = set(indexed_files.keys()) - current_files
            
            if deleted_files:
                console.print(f"[red]Found {len(deleted_files)} deleted files that will be removed from the index[/]")
                for deleted_file in deleted_files:
                    console.print(f"[red]Deleted: {deleted_file}[/]")
            
            # If nothing has changed, no need to update
            if not new_or_modified_files and not deleted_files:
                console.print("[green]No changes detected in the codebase. Index is up to date.[/]")
                return
                
            console.print(f"[blue]Processing {len(new_or_modified_files)} new or modified files[/]")
            
            with Progress() as progress:
                # Process new/modified files and generate embeddings
                if new_or_modified_files:
                    file_embeddings, file_documents, file_metadata = await self.parallel_processor.process_files_parallel(
                        files=new_or_modified_files,
                        chunk_size=self.config.get('chunk_size', 20),
                        embedding_client=self.ai_client,
                        batch_size=self.config.get('parallel', {}).get('batch_size', 10),
                        progress=progress
                    )
                    
                    if not file_embeddings and new_or_modified_files:
                        console.print("[yellow]Warning: No embeddings were generated for new/modified files.[/]")
                    
                    # Update the existing index with new embeddings
                    if file_embeddings:
                        console.print(f"[blue]Adding {len(file_embeddings)} new embeddings to the index[/]")
                        
                        # Add new embeddings to FAISS index
                        embeddings_array = np.array(file_embeddings).astype('float32')
                        self.faiss_index.add(embeddings_array)
                        
                        # Update documents and metadata
                        self.documents.extend(file_documents)
                        self.metadata.extend(file_metadata)
                
                # Handle deleted files if any
                if deleted_files:
                    console.print("[blue]Removing deleted files from the index...[/]")
                    
                    # Identify indices of chunks to keep
                    keep_indices = []
                    new_documents = []
                    new_metadata = []
                    
                    for i, meta in enumerate(self.metadata):
                        if 'file' in meta and meta['file'] not in deleted_files:
                            keep_indices.append(i)
                            new_metadata.append(meta)
                            new_documents.append(self.documents[i])
                    
                    # Create a new FAISS index with only the kept embeddings
                    dimension = self.faiss_index.d
                    new_index = faiss.IndexFlatL2(dimension)
                    
                    # Extract embeddings to keep from the original index
                    for i in keep_indices:
                        # We need to get the embedding for this document
                        # Since FAISS doesn't support direct extraction, we reconstruct by querying
                        # This is expensive but necessary for removing entries
                        if i < len(self.documents):
                            # Use a dummy search to get the embedding vector
                            D, I = self.faiss_index.search(np.array([self.faiss_index.reconstruct(i)]).astype('float32'), 1)
                            new_index.add(np.array([self.faiss_index.reconstruct(i)]).astype('float32'))
                    
                    # Replace the old index and data
                    self.faiss_index = new_index
                    self.documents = new_documents
                    self.metadata = new_metadata
                    
                    console.print(f"[green]Removed {len(deleted_files)} deleted files from index[/]")
                
                # Update the codebase summary
                if self.config.get('generate_summary', False):
                    task_summary = progress.add_task("[blue]Updating codebase summary...", total=1)
                    console.print("[blue]Updating codebase summary...[/]")
                    await self._generate_codebase_summary(all_files)
                    progress.update(task_summary, advance=1)
                else:
                    # Just create a minimal summary
                    self.codebase_summary = f"# Codebase Summary\n\nTotal files: {len(all_files)}\nFile types: {', '.join(f'{ext} ({count})' for ext, count in sorted(file_extensions.items(), key=lambda x: x[1], reverse=True))}"
                
                # Save state
                task_save = progress.add_task("[blue]Saving updated index...", total=1)
                console.print("[blue]Saving updated index...[/]")
                self.save_state()
                progress.update(task_save, advance=1)
                
                console.print(f"[bold green]Index refresh complete! Current index has {len(self.documents)} chunks from {len(current_files)} files.[/]")
                
        except Exception as e:
            console.print(f"[bold red]Error refreshing index: {str(e)}[/]")
            traceback.print_exc()
            raise

    async def complete_code(self, code_prefix: str, file_path: str = None, line_number: int = None) -> str:
        """Provide intelligent code completion suggestions using full codebase context."""
        try:
            # Get file context and metadata
            file_context = {}
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        file_content = f.read()
                        console.print("[yellow]Warning: Some characters in the file could not be decoded properly.[/]")
                file_context = self._create_direct_file_context(file_path, file_content)
                    
                # Find relevant chunks from the codebase
                file_query = f"Find code related to {file_path} and functions or classes used in it"
                relevant_chunks = await self._get_relevant_chunks(file_query, chunk_count=5)
                
                # Extract imports and dependencies
                structure = extract_code_structure(file_content, Path(file_path).suffix.lstrip('.'))
                imports = structure.imports if structure else []
                
                # Get type information and function signatures
                type_info = self._extract_type_information(file_content, imports)
            
            # Build enhanced prompt with full context
            prompt_parts = [
                "You are an expert programmer providing intelligent code completion.",
                "Complete the code in a way that follows the project's patterns and best practices.",
                f"Current file: {file_path}" if file_path else "",
                f"Line number: {line_number}" if line_number else "",
                "\nProject context:",
                self._format_project_context(),
                "\nRelevant code from other files:",
                self._format_chunks(relevant_chunks) if file_path else "",
                "\nImports and dependencies:",
                "\n".join(imports) if file_path else "",
                "\nType information:",
                type_info if file_path else "",
                "\nCode to complete:",
                code_prefix,
                "\nProvide a completion that:",
                "1. Matches the project's coding style and patterns",
                "2. Uses correct imports and type annotations",
                "3. Follows best practices for error handling",
                "4. Maintains consistency with existing code",
                "\nReturn ONLY the completion code, no explanations."
            ]
            
            prompt = "\n".join(filter(None, prompt_parts))
            
            completion = await self.ai_client.get_completion(
                prompt,
                temperature=0.2,
                max_tokens=500
            )
            
            return completion.strip()
            
        except Exception as e:
            console.print(f"[bold red]Error in code completion: {str(e)}[/]")
            return ""

    async def suggest_inline(self, file_path: str, line_number: int, line_content: str) -> List[Dict[str, any]]:
        """Provide intelligent inline suggestions using full codebase context."""
        try:
            # Read the entire file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    file_content = f.read()
                    console.print("[yellow]Warning: Some characters in the file could not be decoded properly.[/]")
                
            # Create file context
            file_context = self._create_direct_file_context(file_path, file_content)
            
            # Find relevant code chunks
            query = f"Find code patterns and examples similar to: {line_content}"
            relevant_chunks = await self._get_relevant_chunks(query, chunk_count=5)
            
            # Extract code structure
            structure = extract_code_structure(file_content, Path(file_path).suffix.lstrip('.'))
            current_scope = self._get_current_scope(file_content, line_number)
            
            # Build enhanced prompt
            prompt_parts = [
                "You are an expert programmer providing intelligent code suggestions.",
                "Provide 3 alternative ways to write or improve the current line.",
                "Each suggestion should be complete, correct, and follow the project's patterns.",
                f"File: {file_path}",
                f"Line {line_number}: {line_content}",
                "\nCurrent scope:",
                current_scope,
                "\nProject context:",
                self._format_project_context(),
                "\nSimilar patterns found:",
                self._format_chunks(relevant_chunks),
                "\nProvide exactly 3 suggestions that:",
                "1. Match the project's coding style",
                "2. Use appropriate error handling",
                "3. Follow type safety and best practices",
                "4. Consider the current scope and context",
                "\nReturn ONLY the suggestions, one per line."
            ]
            
            prompt = "\n".join(filter(None, prompt_parts))
            
            suggestions_text = await self.ai_client.get_completion(
                prompt,
                temperature=0.3,
                max_tokens=200
            )
            
            # Process and score suggestions
            suggestions = []
            for i, suggestion in enumerate(suggestions_text.strip().split('\n')[:3], 1):
                if suggestion.strip():
                    score = self._calculate_enhanced_suggestion_score(
                        suggestion.strip(),
                        line_content,
                        file_content,
                        relevant_chunks
                    )
                    
                    suggestions.append({
                        "text": suggestion.strip(),
                        "score": score,
                        "type": "inline_suggestion",
                        "id": f"suggestion_{i}",
                        "context": {
                            "file": file_path,
                            "line": line_number,
                            "scope": current_scope
                        }
                    })
            
            # Sort by score and return
            return sorted(suggestions, key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            console.print(f"[bold red]Error generating suggestions: {str(e)}[/]")
            return []

    def _calculate_enhanced_suggestion_score(self, suggestion: str, current_line: str, file_content: str, relevant_chunks: List[SemanticChunk]) -> float:
        """Calculate an enhanced relevance score for a suggestion using multiple factors."""
        score = 0.0
        
        # Basic similarity (20%)
        if current_line:
            common_chars = sum(1 for c in suggestion if c in current_line)
            score += 0.2 * (common_chars / max(len(suggestion), len(current_line)))
        
        # Code style consistency (30%)
        style_score = 0.0
        
        # Indentation matching
        if current_line:
            curr_indent = len(current_line) - len(current_line.lstrip())
            sugg_indent = len(suggestion) - len(suggestion.lstrip())
            if curr_indent == sugg_indent:
                style_score += 0.1
        
        # Project patterns matching
        pattern_matches = 0
        for chunk in relevant_chunks:
            if any(line.strip() in chunk.content for line in suggestion.split('\n')):
                pattern_matches += 1
        style_score += 0.1 * min(1.0, pattern_matches / len(relevant_chunks))
        
        # Syntax completeness
        if suggestion.count('(') == suggestion.count(')') and \
           suggestion.count('{') == suggestion.count('}') and \
           suggestion.count('[') == suggestion.count(']'):
            style_score += 0.1
        
        score += 0.3 * style_score
        
        # Semantic relevance to codebase (50%)
        semantic_score = 0.0
        
        # Check if suggestion uses variables/functions from current scope
        file_tokens = set(re.findall(r'\b\w+\b', file_content))
        suggestion_tokens = set(re.findall(r'\b\w+\b', suggestion))
        common_tokens = suggestion_tokens.intersection(file_tokens)
        
        if suggestion_tokens:
            semantic_score += 0.2 * (len(common_tokens) / len(suggestion_tokens))
        
        # Check relevance to similar patterns
        chunk_relevance = 0.0
        for chunk in relevant_chunks:
            chunk_tokens = set(re.findall(r'\b\w+\b', chunk.content))
            common_with_chunk = suggestion_tokens.intersection(chunk_tokens)
            if suggestion_tokens:
                chunk_relevance = max(chunk_relevance, len(common_with_chunk) / len(suggestion_tokens))
        
        semantic_score += 0.3 * chunk_relevance
        
        score += 0.5 * semantic_score
        
        return min(1.0, score)

    def _get_current_scope(self, file_content: str, line_number: int) -> str:
        """Extract the current code scope at a given line number."""
        lines = file_content.split('\n')
        scope_lines = lines[:line_number]
        
        current_class = None
        current_function = None
        current_block = []
        
        for line in scope_lines:
            # Track class definitions
            class_match = re.match(r'\s*class\s+(\w+)', line)
            if class_match:
                current_class = class_match.group(1)
                
            # Track function definitions
            func_match = re.match(r'\s*(?:async\s+)?def\s+(\w+)', line)
            if func_match:
                current_function = func_match.group(1)
                
            # Track current block
            if line.strip().endswith(':'):
                current_block.append(line.strip())
            elif line.strip() and not line.startswith(' '):
                current_block = []
        
        scope_parts = []
        if current_class:
            scope_parts.append(f"Class: {current_class}")
        if current_function:
            scope_parts.append(f"Function: {current_function}")
        if current_block:
            scope_parts.append(f"Block: {' > '.join(current_block)}")
            
        return "\n".join(scope_parts)

    def _extract_type_information(self, content: str, imports: List[str]) -> str:
        """Extract type information and function signatures from code."""
        type_info = []
        
        # Extract type annotations
        type_annotations = re.findall(r'(?m)^\s*(?:async\s+)?def\s+(\w+)\s*\((.*?)\)\s*->\s*([^:]+):', content)
        for func, params, return_type in type_annotations:
            type_info.append(f"Function: {func}")
            type_info.append(f"Parameters: {params}")
            type_info.append(f"Returns: {return_type}\n")
        
        # Extract class structure
        class_matches = re.finditer(r'(?m)^\s*class\s+(\w+)(?:\((.*?)\))?\s*:', content)
        for match in class_matches:
            class_name = match.group(1)
            base_classes = match.group(2) or ''
            type_info.append(f"Class: {class_name}")
            if base_classes:
                type_info.append(f"Inherits: {base_classes}\n")
        
        # Extract variable type hints
        var_types = re.findall(r'(?m)^\s*(\w+)\s*:\s*([^=\n]+)(?:\s*=|$)', content)
        if var_types:
            type_info.append("Variables:")
            for var, type_hint in var_types:
                type_info.append(f"  {var}: {type_hint}")
        
        return "\n".join(type_info)

    def _format_project_context(self) -> str:
        """Format relevant project-wide context information."""
        context_parts = []
        
        # Add project structure overview
        if hasattr(self, 'project_summary'):
            context_parts.append("Project Overview:")
            context_parts.append(self.project_summary)
        
        # Add common patterns found in codebase
        if hasattr(self, 'common_patterns'):
            context_parts.append("\nCommon Patterns:")
            for pattern in self.common_patterns:
                context_parts.append(f"- {pattern}")
        
        # Add dependency information
        if hasattr(self, 'project_dependencies'):
            context_parts.append("\nKey Dependencies:")
            for dep in self.project_dependencies:
                context_parts.append(f"- {dep}")
        
        return "\n".join(context_parts)

    async def _get_relevant_chunks(self, query: str, chunk_count: int) -> List[SemanticChunk]:
        """Retrieve relevant chunks from the codebase based on a query."""
        try:
            # Get question embedding
            question_embedding = await self.ai_client.get_embeddings([query])
            question_vector = np.array(question_embedding).astype('float32')

            # Search for relevant chunks
            D, I = self.faiss_index.search(question_vector, chunk_count)
            
            # Retrieve relevant chunks and ensure they are SemanticChunk objects
            relevant_chunks = []
            for idx in I[0]:
                if idx < len(self.documents):
                    chunk = self.documents[idx]
                    # If it's already a SemanticChunk, use it as is
                    if isinstance(chunk, SemanticChunk):
                        relevant_chunks.append(chunk)
                    # If it's a raw string, create a SemanticChunk with metadata
                    else:
                        metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                        relevant_chunks.append(SemanticChunk(chunk, metadata))
            
            return relevant_chunks
        
        except Exception as e:
            console.print(f"[bold red]Error retrieving relevant chunks: {str(e)}[/]")
            return []

    async def explain_code(self, code: str, context: str = None, detail_level: str = "medium") -> str:
        """Explain code in a natural, conversational way."""
        prompt_parts = [
            "You are an expert programmer explaining code to a fellow developer.",
            "Explain the code in a natural, conversational way.",
            "Focus on the key concepts and patterns.",
            f"Detail level: {detail_level}",
            "\nCode to explain:",
            code
        ]

        if context:
            prompt_parts.extend([
                "\nAdditional context:",
                context
            ])

        prompt_parts.extend([
            "\nProvide your explanation in a conversational way, like you're talking to a colleague.",
            "Include:",
            "1. What the code does at a high level",
            "2. Any interesting patterns or techniques used",
            "3. Potential improvements or considerations",
            "4. Examples of usage if helpful"
        ])

        prompt = "\n".join(filter(None, prompt_parts))
        
        explanation = await self.ai_client.get_completion(
            prompt,
            temperature=0.7,  # Higher temperature for more natural language
            max_tokens=1000
        )
        
        return explanation.strip()

    async def complete_realtime(self, file_path: str, cursor_position: Dict[str, int], current_content: str) -> Dict[str, any]:
        """Provide real-time code completion as you type.
        
        Args:
            file_path: Path to the current file
            cursor_position: Dict containing 'line' and 'column' numbers
            current_content: Current content of the file up to cursor position
            
        Returns:
            Dict containing completion suggestions and metadata
        """
        try:
            # Get the current line and prefix
            lines = current_content.split('\n')
            current_line = lines[cursor_position['line'] - 1] if cursor_position['line'] <= len(lines) else ""
            prefix = current_line[:cursor_position['column']]
            
            # Get file context (previous few lines)
            context_start = max(0, cursor_position['line'] - 5)
            context_lines = lines[context_start:cursor_position['line']]
            context = '\n'.join(context_lines)
            
            prompt_parts = [
                "You are an expert programmer providing real-time code completion.",
                "Complete the current line of code naturally, following the codebase patterns.",
                f"File: {file_path}",
                f"Current position: Line {cursor_position['line']}, Column {cursor_position['column']}",
                "\nContext (previous lines):",
                context,
                "\nCurrent line prefix:",
                prefix,
                "\nProvide completion that would make sense here. Return ONLY the completion text."
            ]
            
            prompt = "\n".join(filter(None, prompt_parts))
            
            completion = await self.ai_client.get_completion(
                prompt,
                temperature=0.2,
                max_tokens=100
            )
            
            return {
                "completion": completion.strip(),
                "range": {
                    "start": {"line": cursor_position['line'], "column": cursor_position['column']},
                    "end": {"line": cursor_position['line'], "column": cursor_position['column'] + len(completion.strip())}
                },
                "source": "ai_completion"
            }
            
        except Exception as e:
            console.print(f"[bold red]Error generating real-time completion: {str(e)}[/]")
            return {"completion": "", "error": str(e)}

    async def suggest_inline_realtime(self, file_path: str, cursor_position: Dict[str, int], current_content: str) -> List[Dict[str, any]]:
        """Provide real-time inline suggestions as you type.
        
        Args:
            file_path: Path to the current file
            cursor_position: Dict containing 'line' and 'column' numbers
            current_content: Current content of the file
            
        Returns:
            List of suggestion objects with completion text and metadata
        """
        try:
            # Get the current line
            lines = current_content.split('\n')
            current_line = lines[cursor_position['line'] - 1] if cursor_position['line'] <= len(lines) else ""
            
            # Get surrounding context
            context_start = max(0, cursor_position['line'] - 5)
            context_end = min(len(lines), cursor_position['line'] + 5)
            context = '\n'.join(lines[context_start:context_end])
            
            prompt_parts = [
                "You are an expert programmer providing inline code suggestions.",
                "Provide exactly 3 alternative ways to write or complete the current line.",
                "Each suggestion should be a complete, working line of code.",
                "Follow the codebase patterns and best practices.",
                "Return ONLY the 3 suggestions, one per line, without any additional text or formatting.",
                f"File: {file_path}",
                f"Current line: {current_line}",
                "\nSurrounding context:",
                context
            ]
            
            prompt = "\n".join(filter(None, prompt_parts))
            
            suggestions_text = await self.ai_client.get_completion(
                prompt,
                temperature=0.3,
                max_tokens=200
            )
            
            # Process suggestions
            suggestions = []
            for i, suggestion in enumerate(suggestions_text.strip().split('\n')[:3], 1):
                if suggestion.strip():
                    suggestions.append({
                        "text": suggestion.strip(),
                        "range": {
                            "start": {"line": cursor_position['line'], "column": 0},
                            "end": {"line": cursor_position['line'], "column": len(current_line)}
                        },
                        "type": "inline_suggestion",
                        "id": f"suggestion_{i}",
                        "score": self._calculate_suggestion_score(suggestion.strip(), current_line, context)
                    })
            
            # Sort suggestions by score
            suggestions.sort(key=lambda x: x['score'], reverse=True)
            return suggestions
            
        except Exception as e:
            console.print(f"[bold red]Error generating inline suggestions: {str(e)}[/]")
            return []
            
    def _calculate_suggestion_score(self, suggestion: str, current_line: str, context: str) -> float:
        """Calculate a relevance score for a suggestion.
        
        Args:
            suggestion: The suggested code line
            current_line: The current line being edited
            context: Surrounding code context
            
        Returns:
            float: Score between 0 and 1, higher is better
        """
        score = 0.0
        
        # Similarity to current line (if it exists)
        if current_line:
            # Basic character-level similarity
            common_chars = sum(1 for c in suggestion if c in current_line)
            score += 0.3 * (common_chars / max(len(suggestion), len(current_line)))
        
        # Context matching
        context_tokens = set(context.split())
        suggestion_tokens = set(suggestion.split())
        if context_tokens:
            matching_tokens = suggestion_tokens.intersection(context_tokens)
            score += 0.3 * (len(matching_tokens) / len(suggestion_tokens) if suggestion_tokens else 0)
        
        # Code style consistency
        style_score = 0.0
        # Indentation matching
        if current_line:
            curr_indent = len(current_line) - len(current_line.lstrip())
            sugg_indent = len(suggestion) - len(suggestion.lstrip())
            if curr_indent == sugg_indent:
                style_score += 0.1
        
        # Common coding patterns
        if suggestion.endswith(';') and context.count(';') > context.count('\n') / 2:
            style_score += 0.1
        if suggestion.endswith(')') and suggestion.count('(') == suggestion.count(')'):
            style_score += 0.1
        
        score += 0.4 * style_score
        
        return min(1.0, score)

    def _format_chunks(self, chunks: List[SemanticChunk]) -> str:
        """Format a list of chunks into a readable string."""
        if not chunks:
            return ""
            
        formatted = []
        for chunk in chunks:
            if not isinstance(chunk, SemanticChunk):
                continue
                
            metadata = chunk.metadata
            formatted.append(f"\nFrom {metadata.get('file')} (lines {metadata.get('start_line')}-{metadata.get('end_line')}):")
            formatted.append("```python")
            formatted.append(chunk.content.strip())
            formatted.append("```\n")
            
        return "\n".join(formatted)

    def _analyze_code_issues(self, content: str, file_path: str) -> List[Dict]:
        """Analyze code for potential issues and bugs.
        
        Args:
            content: The code content to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            List of issues found, each with location, type, and description
        """
        issues = []
        
        try:
            # Parse the code
            tree = ast.parse(content)
            
            # Track variable assignments and usage
            assignments = {}
            undefined_vars = set()
            
            # Track function definitions and calls
            defined_functions = set()
            called_functions = set()
            
            # Track exception handling
            bare_excepts = []
            
            # Visit the AST
            class IssueVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.current_function_params = set()  # Track current function's parameters
                    self.scope_assignments = [{}]  # Stack of variable assignments for each scope
                
                def visit_FunctionDef(self, node):
                    # Add function parameters to current scope
                    self.current_function_params = {arg.arg for arg in node.args.args}
                    defined_functions.add(node.name)
                    # Create new scope for function body
                    self.scope_assignments.append({})
                    self.generic_visit(node)
                    # Remove function scope
                    self.scope_assignments.pop()
                    self.current_function_params = set()
                
                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Store):
                        # Variable assignment - add to current scope
                        self.scope_assignments[-1][node.id] = node.lineno
                    elif isinstance(node.ctx, ast.Load):
                        # Variable usage - check if defined
                        if (node.id not in self.current_function_params and  # Not a function parameter
                            node.id not in __builtins__ and  # Not a builtin
                            not any(node.id in scope for scope in self.scope_assignments) and  # Not in any scope
                            node.id not in defined_functions):  # Not a defined function
                            undefined_vars.add((node.id, node.lineno))
                    self.generic_visit(node)
                
                def visit_Call(self, node):
                    if isinstance(node.func, ast.Name):
                        called_functions.add((node.func.id, node.lineno))
                    self.generic_visit(node)
                
                def visit_Try(self, node):
                    for handler in node.handlers:
                        if handler.type is None:  # bare except
                            bare_excepts.append(handler.lineno)
                    self.generic_visit(node)
            
            visitor = IssueVisitor()
            visitor.visit(tree)
            
            # Report undefined variables
            for var, line in undefined_vars:
                issues.append({
                    'file': file_path,
                    'line': line,
                    'type': 'Error',
                    'description': f'Undefined variable "{var}" used'
                })
            
            # Report undefined functions
            for func, line in called_functions:
                if func not in defined_functions and func not in __builtins__:
                    issues.append({
                        'file': file_path,
                        'line': line,
                        'type': 'Error',
                        'description': f'Call to undefined function "{func}"'
                    })
            
            # Report bare except clauses
            for line in bare_excepts:
                issues.append({
                    'file': file_path,
                    'line': line,
                    'type': 'Warning',
                    'description': 'Bare except clause used - should catch specific exceptions'
                })
            
            # Check for common logic issues
            lines = content.splitlines()
            for i, line in enumerate(lines, 1):
                # Check for hardcoded credentials
                if re.search(r'password\s*=\s*[\'"][^\'"]+[\'"]', line, re.I):
                    issues.append({
                        'file': file_path,
                        'line': i,
                        'type': 'Warning',
                        'description': 'Hardcoded password found'
                    })
                
                # Check for potential infinite loops
                if re.search(r'while\s+True:', line):
                    if not any('break' in l for l in lines[i:i+10]):  # Check next 10 lines
                        issues.append({
                            'file': file_path,
                            'line': i,
                            'type': 'Warning',
                            'description': 'Potential infinite loop - no break statement found'
                        })
            
            return issues
            
        except SyntaxError as e:
            return [{
                'file': file_path,
                'line': e.lineno,
                'type': 'Error',
                'description': f'Syntax error: {str(e)}'
            }]
        except Exception as e:
            return [{
                'file': file_path,
                'line': 1,
                'type': 'Error',
                'description': f'Error analyzing file: {str(e)}'
            }]