# analyzer.py
from typing import List, Dict
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

console = Console()

class CodebaseAnalyzer:
    def __init__(self, path: str, together_api_key: str, project_name: str = None):
        self.path = Path(path)
        self.ai_client = TogetherAIClient(together_api_key)
        self.faiss_index = None
        self.documents = []
        self.metadata = []
        self.conversation_history = []  # Store conversation history
        
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
    
    async def query(self, question: str, chunk_count: int = None) -> str:
        """Query the codebase and get AI response"""
        console.print(f"[bold blue]Querying:[/] {question}")
        
        try:
            # Get question embedding
            console.print("[green]Generating embedding for query...[/]")
            question_embedding = await self.ai_client.get_embeddings([question])
            question_vector = np.array(question_embedding).astype('float32')
            
            # Add keyword search for filenames
            keyword_files = self._keyword_search(question)
            
            # Determine number of chunks to retrieve
            if not chunk_count:
                # Default is 10, but scale based on codebase size
                total_chunks = self.faiss_index.ntotal
                chunk_count = min(max(10, total_chunks // 10), 20)  # Between 10-20 chunks
                
                # Adjust based on question complexity
                if any(term in question.lower() for term in ["overall", "entire", "all", "architecture", "structure"]):
                    chunk_count = min(chunk_count * 2, 30)  # More chunks for broad questions
            
            console.print(f"[green]Searching for top {chunk_count} relevant code chunks...[/]")
            D, I = self.faiss_index.search(question_vector, chunk_count)
            
            # Prepare context from results + keyword results
            console.print("[green]Preparing context from results...[/]")
            context = self._prepare_context(I[0], keyword_files)
            console.print(f"[dim]Using {len(I[0])} relevant code chunks as context[/]")
            
            # Prepare conversation history
            history_context = ""
            if self.conversation_history:
                console.print(f"[green]Including {len(self.conversation_history)} previous exchanges in context[/]")
                history_context = self._prepare_conversation_history()
            
            # Get AI response
            console.print("[green]Generating AI response...[/]")
            response = await self.ai_client.get_completion(question, context, history_context)
            
            # Update conversation history - store before saving
            self.conversation_history.append({"question": question, "answer": response})
            if len(self.conversation_history) > 5:  # Keep last 5 interactions
                self.conversation_history.pop(0)
            
            # Save state to persist conversation history
            self.save_state()
            
            return response
        except Exception as e:
            console.print(f"[bold red]Error during query: {str(e)}[/]")
            traceback.print_exc()
            raise
    
    def save_state(self):
        """Save index and metadata to disk"""
        try:
            state = {
                'documents': self.documents,
                'metadata': self.metadata,
                'conversation_history': self.conversation_history  # Save conversation history
            }
            
            # Save FAISS index
            index_file = str(self.project_dir / "code.index")
            faiss.write_index(self.faiss_index, index_file)
            console.print(f"[green]FAISS index saved to {index_file}[/]")
            
            # Save documents and metadata
            state_file = self.project_dir / "state.pkl"
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
            console.print(f"[green]Metadata and conversation history saved to {state_file}[/]")
        except Exception as e:
            console.print(f"[bold red]Error saving state: {str(e)}[/]")
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
        # Combine semantic and keyword search results
        all_indices = list(indices)
        if keyword_indices:
            all_indices.extend(keyword_indices)
            all_indices = list(set(all_indices))  # Deduplicate
            
        context_parts = []
        
        # First gather all filenames for better navigation
        filenames = set()
        for idx in all_indices:
            if idx < len(self.metadata):
                filenames.add(self.metadata[idx]['file'])
                
        # Add file overview section
        if filenames:
            context_parts.append(f"Files referenced: {', '.join(sorted(filenames))}\n")
            
        # Then add actual code chunks
        for idx in all_indices:
            if idx < len(self.documents):
                doc = self.documents[idx]
                metadata = self.metadata[idx]
                context_parts.append(
                    f"File: {metadata['file']}\n"
                    f"Lines {metadata['start_line']}-{metadata['end_line']}:\n"
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
        """Split file into chunks with overlap"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.readlines()
            
            chunks = []
            chunk_size = 20  # lines per chunk
            overlap = 5  # overlapping lines
            
            for i in range(0, len(content), chunk_size - overlap):
                chunk_lines = content[i:i + chunk_size]
                if chunk_lines:
                    chunks.append({
                        "text": "".join(chunk_lines),
                        "start_line": i + 1,
                        "end_line": i + len(chunk_lines)
                    })
            
            return chunks
        except UnicodeDecodeError:
            console.print(f"[yellow]Warning: Could not process {file_path}: UnicodeDecodeError - probably a binary file[/]")
            return []
        except Exception as e:
            console.print(f"[yellow]Warning: Could not process {file_path}: {str(e)}[/]")
            traceback.print_exc()
            return []