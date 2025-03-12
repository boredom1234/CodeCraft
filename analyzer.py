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
from rich.progress import Progress, TextColumn, BarColumn, TaskID, TimeRemainingColumn, SpinnerColumn
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

console = Console()

class CodeStructure:
    """Helper class to track code structure and context"""
    def __init__(self):
        self.imports = []
        self.classes = {}  # class_name -> {methods: [], attributes: []}
        self.functions = []
        self.global_vars = []
        self.docstrings = {}  # node -> docstring
        self.comments = []
        
    def add_import(self, import_stmt: str):
        self.imports.append(import_stmt.strip())
        
    def add_class(self, class_name: str, methods: List[str], attributes: List[str]):
        self.classes[class_name] = {"methods": methods, "attributes": attributes}
        
    def add_function(self, func_name: str):
        self.functions.append(func_name)
        
    def add_global(self, var_name: str):
        self.global_vars.append(var_name)
        
    def add_docstring(self, node: ast.AST, docstring: str):
        self.docstrings[node] = docstring.strip()
        
    def add_comment(self, comment: str):
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
            "debug": False
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
        """Index the codebase"""
        console.print("[bold blue]Starting indexing process...[/]")
        
        try:
            # Create a rich progress display
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
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
                    console.print(f"[dim]Sample files: {', '.join([str(f) for f in sample_files])}{' ...' if len(all_files) > 5 else ''}[/]")
                
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
                            
                        texts = [chunk["text"] for chunk in chunks]
                        
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
                                "start_line": chunk["start_line"],
                                "end_line": chunk["end_line"]
                            } for chunk in chunks])
                            
                            embedded_files += 1
                            console.print(f"[green]Successfully embedded {file_path} ({len(chunks)} chunks)[/]")
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
        return Panel(
            Text(title, style="bold blue", justify="center"),
            box=box.ROUNDED,
            border_style="blue",
            padding=(1, 2)
        )

    def _create_footer(self, text: str) -> Panel:
        """Create a styled footer panel"""
        return Panel(
            Text(text, style="dim", justify="center"),
            box=box.ROUNDED,
            border_style="blue",
            padding=(1, 2)
        )

    def _format_output(self, text: str, title: str = "Code Analysis"):
        """Format text using Rich's markdown renderer with enhanced layout"""
        try:
            # Create header
            header = self._create_header(title)
            
            # Create main content
            md = Markdown(text)
            main_panel = Panel(
                md,
                border_style="blue",
                box=box.ROUNDED,
                padding=(1, 2)
            )
            
            # Create footer with stats
            footer_text = f"Project: {self.project_name} | Chunks: {len(self.documents)} | Files: {len(set(m['file'] for m in self.metadata))}"
            footer = self._create_footer(footer_text)
            
            # Update layout
            self.layout["header"].update(header)
            self.layout["main"].update(main_panel)
            self.layout["footer"].update(footer)
            
            # Print the layout
            self.console.print(self.layout)
            
        except Exception as e:
            self.console.print(f"[bold red]Error formatting output:[/] {str(e)}")

    async def query(self, question: str, chunk_count: int = None) -> str:
        """Query the codebase and get AI response with enhanced context"""
        header = self._create_header("Code Query")
        self.console.print(header)
        
        try:
            with self.console.status("[bold green]Processing query...", spinner="dots") as status:
                # Enhanced query preprocessing
                query_context = self._analyze_query(question)
                self.console.print(f"[dim]Query context: {query_context}[/]")
                
                # Get question embedding with enhanced context
                status.update("[bold green]Generating embedding...")
                enhanced_question = self._enhance_query_with_context(question, query_context)
                self.console.print(f"[dim]Enhanced question: {enhanced_question}[/]")
                question_embedding = await self.ai_client.get_embeddings([enhanced_question])
                question_vector = np.array(question_embedding).astype('float32')
                
                # Smart chunk count determination
                if not chunk_count:
                    chunk_count = self._determine_optimal_chunk_count(question, query_context)
                self.console.print(f"[dim]Using {chunk_count} chunks[/]")
                
                # Enhanced search with multiple strategies
                status.update("[bold green]Searching codebase...")
                relevant_chunks = await self._enhanced_search(
                    question_vector,
                    chunk_count,
                    query_context
                )
                
                # Debug print metadata of found chunks
                self.console.print("[dim]Found chunks:[/]")
                for idx in relevant_chunks:
                    if idx < len(self.metadata):
                        meta = self.metadata[idx]
                        self.console.print(f"[dim]  - {meta['file']} (lines {meta['start_line']}-{meta['end_line']})[/]")
                
                # Prepare enhanced context
                status.update("[bold green]Preparing context...")
                context = self._prepare_enhanced_context(relevant_chunks, query_context)
                
                # Include relevant conversation history
                history_context = ""
                if self.conversation_history:
                    status.update("[bold green]Including conversation history...")
                    history_context = self._prepare_filtered_history(question, query_context)
                
                # Generate response with enhanced prompt
                status.update("[bold green]Generating response...")
                response = await self._generate_enhanced_response(
                    question,
                    context,
                    history_context,
                    query_context
                )
                
                # Update conversation history with context
                self._update_conversation_history(question, response, query_context)
                
                # Format and display the response
                self._format_output(response, title="AI Response")
                
                return response
                
        except Exception as e:
            self.console.print(Panel(
                Text(f"Error: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
                box=box.ROUNDED
            ))
            traceback.print_exc()
            raise
    
    def save_state(self):
        """Save index and metadata to disk"""
        try:
            state = {
                'documents': self.documents,
                'metadata': self.metadata,
                'conversation_history': self.conversation_history
            }
            
            # Save FAISS index
            index_file = str(self.project_dir / "code.index")
            faiss.write_index(self.faiss_index, index_file)
            self.console.print("[green]✓[/] FAISS index saved")
            
            # Save documents and metadata
            state_file = self.project_dir / "state.pkl"
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
            self.console.print("[green]✓[/] Metadata and history saved")
            
        except Exception as e:
            self.console.print(f"[bold red]Error saving state:[/] {str(e)}")
            traceback.print_exc()
            raise
    
    def load_state(self):
        """Load index and metadata from disk"""
        try:
            # Load FAISS index
            index_file = str(self.project_dir / "code.index")
            self.faiss_index = faiss.read_index(index_file)
            console.print(f"[green]FAISS index loaded from {index_file}[/]")
            
            # Load documents and metadata
            state_file = self.project_dir / "state.pkl"
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
                self.documents = state['documents']
                self.metadata = state['metadata']
                if 'conversation_history' in state:
                    self.conversation_history = state['conversation_history']
            console.print(f"[green]Metadata loaded from {state_file}[/]")
            console.print(f"[green]Loaded {len(self.documents)} document chunks and {len(self.metadata)} metadata records[/]")
            console.print(f"[green]Project: {self.project_name}[/]")
            if hasattr(self, 'conversation_history') and self.conversation_history:
                console.print(f"[green]Loaded {len(self.conversation_history)} conversation exchanges[/]")
        except Exception as e:
            console.print(f"[bold red]Error loading state: {str(e)}[/]")
            traceback.print_exc()
            raise
    
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
    
    def _prepare_enhanced_context(self, relevant_chunks: List[int], query_context: Dict) -> str:
        """Prepare enhanced context from relevant chunks"""
        context_parts = []
        for idx in relevant_chunks:
            if idx < len(self.documents):
                doc = self.documents[idx]
                meta = self.metadata[idx]
                
                # Add file information
                context_parts.append(f"File: {meta['file']}")
                context_parts.append(f"Lines {meta['start_line']}-{meta['end_line']}")
                
                # Add the actual code content
                context_parts.append("```" + meta['file'].split('.')[-1])
                context_parts.append(doc)
                context_parts.append("```")
                context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)

    def _prepare_context(self, indices, keyword_indices=None) -> str:
        """Prepare context from search results"""
        try:
            all_indices = list(indices)
            if keyword_indices:
                all_indices.extend(keyword_indices)
                all_indices = list(set(all_indices))
            
            context_parts = []
            for idx in all_indices:
                if idx < len(self.metadata):
                    meta = self.metadata[idx]
                    doc = self.documents[idx]
                    
                    # Add file information
                    context_parts.append(f"File: {meta['file']}")
                    context_parts.append(f"Lines {meta['start_line']}-{meta['end_line']}")
                    
                    # Add the actual code content
                    context_parts.append("```" + meta['file'].split('.')[-1])
                    context_parts.append(doc)
                    context_parts.append("```")
                    context_parts.append("")  # Empty line for separation
            
            return "\n".join(context_parts)
            
        except Exception as e:
            # Fallback to basic context format
            basic_parts = []
            for idx in all_indices:
                if idx < len(self.metadata):
                    meta = self.metadata[idx]
                    doc = self.documents[idx]
                    basic_parts.append(
                        f"File: {meta['file']}\n"
                        f"Lines {meta['start_line']}-{meta['end_line']}:\n"
                        f"{doc}\n"
                    )
            return "\n---\n".join(basic_parts)
    
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
    
    def _chunk_file(self, file_path: Path) -> List[Dict]:
        """Split file into chunks with intelligent parsing and context"""
        try:
            # Read the entire file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines(keepends=True)
            
            # Determine file type for language-specific parsing
            file_type = file_path.suffix.lstrip('.')
            
            # Extract code structure with language-specific parsing
            structure = extract_code_structure(content, file_type)
            
            chunks = []
            chunk_size = self.config.get('chunk_size', 20)
            overlap = self.config.get('overlap', 5)
            
            # Enhanced chunking with semantic boundaries
            def find_semantic_boundary(lines: List[str], start: int, end: int, file_type: str) -> int:
                """Find the best semantic boundary based on language-specific patterns"""
                # Language-specific patterns with priorities
                patterns = {
                    'py': {
                        'class ': 10,
                        'def ': 9,
                        'async def ': 9,
                        '\n\n': 8,
                        '    def ': 7,
                        '    async def ': 7,
                        'if __name__': 6,
                        'return ': 5
                    },
                    'go': {
                        'type ': 10,
                        'func ': 9,
                        '\n\n': 8,
                        'struct {': 7,
                        'interface {': 7,
                        'return ': 5
                    },
                    'js': {
                        'class ': 10,
                        'function ': 9,
                        'const ': 8,
                        'let ': 8,
                        'return ': 5
                    },
                    'ts': {
                        'interface ': 10,
                        'class ': 10,
                        'function ': 9,
                        'const ': 8,
                        'let ': 8,
                        'return ': 5
                    }
                }
                
                # Common patterns for all languages
                common_patterns = {
                    '}\n': 4,
                    ']\n': 3,
                    ')\n': 2,
                    ';\n': 1
                }
                
                # Get language-specific patterns or fall back to common ones
                priority = patterns.get(file_type, {})
                priority.update(common_patterns)
                
                best_score = -1
                best_pos = end
                
                for i in range(start, end):
                    line = lines[i]
                    for pattern, score in priority.items():
                        if line.strip().startswith(pattern) or line.endswith(pattern):
                            if score > best_score:
                                best_score = score
                                best_pos = i
                
                return best_pos if best_score > -1 else end
            
            # Enhanced context tracking
            context_stack = []
            current_class = None
            current_function = None
            imports_section = []
            
            with Progress() as progress:
                task = progress.add_task(
                    f"[green]Chunking {file_path.name}...", 
                    total=len(lines)
                )
                
                current_line = 1
                i = 0
                while i < len(lines):
                    # Find semantic boundary for chunk end
                    chunk_end = min(i + chunk_size, len(lines))
                    semantic_end = find_semantic_boundary(lines, i + chunk_size - overlap, chunk_end, file_type)
                    chunk_lines = lines[i:semantic_end]
                    
                    if chunk_lines:
                        # Track context
                        chunk_context = {
                            'imports': [],
                            'classes': [],
                            'functions': [],
                            'scope': [],
                            'dependencies': set()
                        }
                        
                        # Analyze chunk content
                        chunk_content = ''.join(chunk_lines)
                        try:
                            tree = ast.parse(chunk_content)
                            for node in ast.walk(tree):
                                if isinstance(node, ast.Import):
                                    for name in node.names:
                                        chunk_context['imports'].append(name.name)
                                        chunk_context['dependencies'].add(name.name)
                                elif isinstance(node, ast.ImportFrom):
                                    module = node.module or ''
                                    for name in node.names:
                                        import_stmt = f"from {module} import {name.name}"
                                        chunk_context['imports'].append(import_stmt)
                                        chunk_context['dependencies'].add(f"{module}.{name.name}")
                                elif isinstance(node, ast.ClassDef):
                                    chunk_context['classes'].append(node.name)
                                    chunk_context['scope'].append(f"class {node.name}")
                                elif isinstance(node, ast.FunctionDef):
                                    chunk_context['functions'].append(node.name)
                                    chunk_context['scope'].append(f"function {node.name}")
                        except:
                            pass  # Skip AST parsing if chunk is incomplete
                        
                        # Get the chunk with enhanced context
                        chunk_text = format_chunk_with_context(
                            chunk_lines,
                            structure,
                            current_line,
                            str(file_path.relative_to(self.path))
                        )
                        
                        # Add semantic metadata
                        chunks.append({
                            "text": chunk_text,
                            "start_line": current_line,
                            "end_line": current_line + len(chunk_lines) - 1,
                            "imports": list(set(structure.imports + chunk_context['imports'])),
                            "classes": list(set(list(structure.classes.keys()) + chunk_context['classes'])),
                            "functions": list(set(structure.functions + chunk_context['functions'])),
                            "has_docstring": bool(structure.docstrings),
                            "file_type": file_path.suffix,
                            "scope": chunk_context['scope'],
                            "dependencies": list(chunk_context['dependencies']),
                            "semantic_context": {
                                "parent_class": current_class,
                                "parent_function": current_function,
                                "is_method": bool(current_class and current_function),
                                "is_nested": len(chunk_context['scope']) > 1
                            }
                        })
                        
                        # Update line counter and progress
                        current_line = current_line + len(chunk_lines)
                        i = semantic_end
                        progress.update(task, advance=len(chunk_lines))
                    else:
                        i += 1
            
            return chunks
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not process {file_path}: {str(e)}[/]")
            traceback.print_exc()
            return []

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
        
        return context

    def _enhance_query_with_context(self, question: str, context: Dict) -> str:
        """Enhance query with extracted context for better embedding"""
        enhanced_parts = [question]
        
        # Add type-specific context
        type_contexts = {
            'explain': 'Explain and describe in detail',
            'find': 'Search and locate specific elements',
            'modify': 'Modify or update existing code',
            'create': 'Create or implement new functionality',
            'error': 'Debug and fix issues'
        }
        
        if context['type'] in type_contexts:
            enhanced_parts.append(type_contexts[context['type']])
        
        # Add code elements context
        for element_type, elements in context['code_elements'].items():
            if elements:
                enhanced_parts.append(f"{element_type}: {', '.join(elements)}")
        
        # Add focus context
        if context['focus']:
            focus_str = ', '.join(f"{f_type} {f_name}" for f_type, f_name in context['focus'])
            enhanced_parts.append(f"Focus on: {focus_str}")
        
        return ' | '.join(enhanced_parts)

    async def _enhanced_search(self, query_vector: np.ndarray, chunk_count: int, 
                             query_context: Dict) -> List[Dict]:
        """Perform enhanced search using multiple strategies"""
        # Get initial semantic search results
        D, I = self.faiss_index.search(query_vector, chunk_count * 2)  # Get more candidates
        
        # Score and filter results
        scored_results = []
        for idx in I[0]:
            if idx < len(self.metadata):
                score = self._score_chunk_relevance(
                    self.metadata[idx],
                    self.documents[idx],
                    query_context
                )
                scored_results.append((score, idx))
        
        # Sort by relevance score and take top chunks
        scored_results.sort(reverse=True)
        top_indices = [idx for _, idx in scored_results[:chunk_count]]
        
        return top_indices

    def _score_chunk_relevance(self, metadata: Dict, content: str, 
                             query_context: Dict) -> float:
        """Score chunk relevance based on multiple factors"""
        score = 0.0
        
        # Context match score
        if query_context['focus']:
            for focus_type, focus_name in query_context['focus']:
                if focus_type == 'file' and focus_name in metadata['file']:
                    score += 2.0
                elif focus_type in ['class', 'function', 'method']:
                    if focus_name in content:
                        score += 1.5
        
        # Code elements match score
        for element_type, elements in query_context['code_elements'].items():
            for element in elements:
                if element in content:
                    score += 1.0
        
        # Scope relevance
        if metadata.get('scope'):
            scope_relevance = sum(
                2.0 if any(f_name in s for _, f_name in query_context['focus'])
                else 0.5
                for s in metadata['scope']
            )
            score += scope_relevance
        
        # Dependency relevance
        if metadata.get('dependencies'):
            dep_relevance = sum(
                1.0 if dep in query_context['code_elements']['imports']
                else 0.5
                for dep in metadata['dependencies']
            )
            score += dep_relevance
        
        return score

    def _determine_optimal_chunk_count(self, question: str, context: Dict) -> int:
        """Determine optimal number of chunks based on query complexity"""
        base_count = 10
        
        # Adjust for query type
        type_multipliers = {
            'explain': 1.5,  # Need more context for explanations
            'find': 1.0,     # Standard search
            'modify': 1.2,   # Need surrounding context
            'create': 1.3,   # Need examples and related code
            'error': 1.4     # Need more context for debugging
        }
        
        multiplier = type_multipliers.get(context['type'], 1.0)
        
        # Adjust for complexity
        complexity_score = (
            len(context['focus']) * 0.3 +
            sum(len(elements) for elements in context['code_elements'].values()) * 0.2 +
            (2.0 if any(term in question.lower() for term in 
                       ["overall", "entire", "all", "architecture", "structure"]) else 0)
        )
        
        chunk_count = int(base_count * multiplier * (1 + complexity_score))
        return min(max(chunk_count, 5), 30)  # Keep within reasonable bounds

    async def _generate_enhanced_response(self, question: str, context: str,
                                       history_context: str, query_context: Dict) -> str:
        """Generate response with enhanced prompt engineering"""
        # Build enhanced prompt
        prompt_parts = [
            f"Question type: {query_context['type']}",
            f"Focus areas: {', '.join(f'{t}: {n}' for t, n in query_context['focus'])}" if query_context['focus'] else "",
            "Code elements mentioned:",
            *[f"- {k}: {', '.join(v)}" for k, v in query_context['code_elements'].items() if v],
            "\nQuestion:",
            question,
            "\nRelevant code context:",
            context
        ]
        
        if history_context:
            prompt_parts.extend([
                "\nRelevant conversation history:",
                history_context
            ])
        
        enhanced_prompt = "\n".join(filter(None, prompt_parts))
        
        # Generate response with enhanced prompt
        return await self.ai_client.get_completion(
            enhanced_prompt,
            temperature=self.config.get('temperature', 0.7)
        )

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
        relevant_history = [
            exchange for exchange in self.conversation_history
            if any(term in exchange['question'].lower() for term in query_context['keywords'])
        ]
        return self._prepare_conversation_history()