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

def extract_code_structure(content: str) -> CodeStructure:
    """Extract structural information from Python code"""
    structure = CodeStructure()
    
    try:
        # Parse the AST
        tree = ast.parse(content)
        
        # Extract docstrings and comments
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    structure.add_docstring(node, docstring)
        
        # Extract comments using regex (since they're not in AST)
        comment_pattern = r'#.*$'
        for line in content.split('\n'):
            comment_match = re.search(comment_pattern, line)
            if comment_match:
                structure.add_comment(comment_match.group())
        
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
                
                # Extract methods and attributes
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
                        
        return structure
    except Exception as e:
        console.print(f"[yellow]Warning: Could not parse code structure: {str(e)}[/]")
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
        
        # Initialize Rich console with markdown support
        self.console = Console(force_terminal=True)
        
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
    
    def _format_output(self, text: str):
        """Format text using Rich's markdown renderer"""
        # Create a markdown object
        md = Markdown(text)
        # Render in a nice panel
        panel = Panel(
            md,
            title="Code Analysis",
            border_style="blue",
            box=box.ROUNDED
        )
        self.console.print(panel)

    async def query(self, question: str, chunk_count: int = None) -> str:
        """Query the codebase and get AI response"""
        self.console.print(f"[bold blue]Querying:[/] {question}")
        
        try:
            # Get question embedding
            self.console.print("[green]Generating embedding for query...[/]")
            question_embedding = await self.ai_client.get_embeddings([question])
            question_vector = np.array(question_embedding).astype('float32')
            
            # Add keyword search for filenames
            keyword_files = self._keyword_search(question)
            
            # Determine number of chunks to retrieve
            if not chunk_count:
                total_chunks = self.faiss_index.ntotal
                chunk_count = min(max(10, total_chunks // 10), 20)
                
                if any(term in question.lower() for term in ["overall", "entire", "all", "architecture", "structure"]):
                    chunk_count = min(chunk_count * 2, 30)
            
            with self.console.status("[bold green]Searching codebase...", spinner="dots"):
                D, I = self.faiss_index.search(question_vector, chunk_count)
            
            # Prepare context from results + keyword results
            with self.console.status("[bold green]Preparing context...", spinner="dots"):
                context = self._prepare_context(I[0], keyword_files)
            
            # Prepare conversation history
            history_context = ""
            if self.conversation_history:
                with self.console.status("[bold green]Including conversation history...", spinner="dots"):
                    history_context = self._prepare_conversation_history()
            
            # Get AI response
            with self.console.status("[bold green]Generating response...", spinner="dots"):
                response = await self.ai_client.get_completion(question, context, history_context)
            
            # Update conversation history
            self.conversation_history.append({"question": question, "answer": response})
            if len(self.conversation_history) > 5:
                self.conversation_history.pop(0)
            
            # Save state
            with self.console.status("[bold green]Saving state...", spinner="dots"):
                self.save_state()
            
            # Format and display the response
            self._format_output(response)
            
            return response
            
        except Exception as e:
            self.console.print(f"[bold red]Error during query:[/] {str(e)}")
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
    
    def _prepare_context(self, indices, keyword_indices=None) -> str:
        """Prepare context from search results"""
        all_indices = list(indices)
        if keyword_indices:
            all_indices.extend(keyword_indices)
            all_indices = list(set(all_indices))
            
        # Create a table for file overview
        table = Table(title="Files Referenced", box=box.ROUNDED)
        table.add_column("File", style="cyan")
        table.add_column("Lines", style="green")
        
        context_parts = []
        filenames = set()
        
        for idx in all_indices:
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                filenames.add(meta['file'])
                table.add_row(
                    meta['file'],
                    f"{meta['start_line']}-{meta['end_line']}"
                )
        
        if filenames:
            self.console.print(table)
            
        # Add code chunks with syntax highlighting
        for idx in all_indices:
            if idx < len(self.documents):
                doc = self.documents[idx]
                meta = self.metadata[idx]
                
                # Create syntax highlighted code
                syntax = Syntax(
                    doc,
                    "python",
                    line_numbers=True,
                    start_line=meta['start_line']
                )
                
                context_parts.append(
                    f"File: {meta['file']}\n"
                    f"Lines {meta['start_line']}-{meta['end_line']}:\n"
                    f"{doc}\n"
                )
        
        return "\n---\n".join(context_parts)
    
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
            
            # Extract code structure
            structure = extract_code_structure(content)
            
            chunks = []
            chunk_size = self.config.get('chunk_size', 20)
            overlap = self.config.get('overlap', 5)
            
            with Progress() as progress:
                task = progress.add_task(
                    f"[green]Chunking {file_path.name}...", 
                    total=len(lines)
                )
                
                current_line = 1
                for i in range(0, len(lines), chunk_size - overlap):
                    chunk_lines = lines[i:i + chunk_size]
                    if chunk_lines:
                        # Get the chunk with context
                        chunk_text = format_chunk_with_context(
                            chunk_lines,
                            structure,
                            current_line,
                            str(file_path.relative_to(self.path))
                        )
                        
                        chunks.append({
                            "text": chunk_text,
                            "start_line": current_line,
                            "end_line": current_line + len(chunk_lines) - 1,
                            "imports": structure.imports,
                            "classes": list(structure.classes.keys()),
                            "functions": structure.functions,
                            "has_docstring": bool(structure.docstrings),
                            "file_type": file_path.suffix
                        })
                        current_line = current_line + (chunk_size - overlap)
                    progress.update(task, advance=len(chunk_lines))
            
            return chunks
        except Exception as e:
            console.print(f"[yellow]Warning: Could not process {file_path}: {str(e)}[/]")
            traceback.print_exc()
            return []