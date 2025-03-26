import asyncio
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
from dataclasses import dataclass
from rich.progress import Progress, TaskID
import traceback
from rich.console import Console
import ast
import re

console = Console()

@dataclass
class ProcessingResult:
    """Result from processing a single file"""
    file_path: str
    chunks: List[Dict]
    embeddings: Optional[List[List[float]]] = None
    error: Optional[str] = None

def process_file(file_path: str, chunk_size: int = 20) -> List[Dict]:
    """Process a single file and return chunks. This function runs in a separate process."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lines = content.splitlines()
        chunks = []
        i = 0
        
        while i < len(lines):
            # Skip empty lines
            if not lines[i].strip():
                i += 1
                continue
                
            # Determine chunk type
            if re.match(r'^\s*class\s+', lines[i]):
                chunk_type = 'class_definition'
            elif re.match(r'^\s*(async\s+)?def\s+', lines[i]):
                chunk_type = 'function_definition'
            elif re.match(r'^\s*(import|from)\s+\w+', lines[i]):
                chunk_type = 'import_section'
            else:
                chunk_type = 'code_block'
                
            # Find chunk boundaries
            chunk_start = i
            chunk_end = find_chunk_boundary(lines, i, chunk_type, chunk_size)
            
            # Extract chunk content
            chunk_lines = lines[chunk_start:chunk_end]
            chunk_content = '\n'.join(chunk_lines)
            
            # Create chunk metadata
            metadata = {
                'file': str(file_path),
                'start_line': chunk_start + 1,
                'end_line': chunk_end,
                'type': chunk_type,
                'has_docstring': bool(re.search(r'""".*?"""', chunk_content, re.DOTALL)),
                'has_comments': bool(re.search(r'#.*$', chunk_content, re.MULTILINE))
            }
            
            chunks.append({
                'content': chunk_content,
                'metadata': metadata
            })
            
            i = chunk_end
            
        return chunks
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not process {file_path}: {str(e)}[/]")
        return []

def find_chunk_boundary(lines: List[str], start: int, chunk_type: str, chunk_size: int) -> int:
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

class ParallelProcessor:
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize parallel processor with optional worker count"""
        self.max_workers = max_workers or mp.cpu_count()
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
    async def process_files_parallel(self, 
                                   files: List[Path],
                                   chunk_size: int = 20,
                                   embedding_client: Any = None,
                                   batch_size: int = 10,
                                   progress: Optional[Progress] = None) -> Tuple[List, List, List]:
        """Process files in parallel with batched embedding generation
        
        Args:
            files: List of files to process
            chunk_size: Size of code chunks
            embedding_client: Client for generating embeddings
            batch_size: Size of batches for embedding generation
            progress: Optional progress tracker
            
        Returns:
            Tuple of (all_embeddings, all_documents, all_metadata)
        """
        try:
            # Create progress tasks if progress tracker provided
            task_process = None
            task_embed = None
            if progress:
                task_process = progress.add_task("[green]Processing files...", total=len(files))
                task_embed = progress.add_task("[yellow]Waiting for files to process...", total=len(files))
            
            # Process files in parallel using process pool
            loop = asyncio.get_event_loop()
            chunk_futures = []
            
            for file_path in files:
                future = loop.run_in_executor(
                    self.process_pool,
                    process_file,
                    str(file_path),
                    chunk_size
                )
                chunk_futures.append(future)
            
            # Collect results as they complete
            results = []
            for i, future in enumerate(asyncio.as_completed(chunk_futures)):
                try:
                    current_file = str(files[i])
                    current_file_name = Path(current_file).name
                    
                    if progress and task_process:
                        progress.update(task_process, description=f"[green]Processing file: {current_file_name}")
                    
                    chunks = await future
                    if chunks:
                        results.append(ProcessingResult(
                            file_path=current_file,
                            chunks=chunks
                        ))
                        console.print(f"[dim]Processed {current_file_name} - {len(chunks)} chunks[/]")
                    if progress and task_process:
                        progress.update(task_process, advance=1)
                except Exception as e:
                    console.print(f"[red]Error processing {files[i]}: {str(e)}[/]")
                    traceback.print_exc()
            
            # Batch process embeddings using thread pool for I/O-bound tasks
            all_embeddings = []
            all_documents = []
            all_metadata = []
            
            for batch_start in range(0, len(results), batch_size):
                batch = results[batch_start:batch_start + batch_size]
                
                # Collect all texts from this batch
                batch_texts = []
                # Get unique file names for this batch for progress display
                batch_files = []
                for result in batch:
                    batch_files.append(Path(result.file_path).name)
                    for chunk in result.chunks:
                        batch_texts.append(chunk['content'])
                
                # Display which files are being embedded
                current_files = ", ".join(batch_files[:3])
                if len(batch_files) > 3:
                    current_files += f" (+{len(batch_files)-3} more)"
                
                if progress and task_embed:
                    progress.update(task_embed, description=f"[yellow]Generating embeddings for: {current_files}")
                
                # Generate embeddings for the batch
                try:
                    if embedding_client:
                        console.print(f"[dim]Embedding batch of {len(batch_files)} files: {current_files}[/]")
                        batch_embeddings = await embedding_client.get_embeddings(batch_texts)
                        
                        # Store results
                        text_idx = 0
                        for result in batch:
                            for chunk in result.chunks:
                                if text_idx < len(batch_embeddings):
                                    all_embeddings.append(batch_embeddings[text_idx])
                                    all_documents.append(chunk['content'])
                                    all_metadata.append(chunk['metadata'])
                                text_idx += 1
                                
                        if progress and task_embed:
                            progress.update(task_embed, advance=len(batch))
                            
                except Exception as e:
                    console.print(f"[red]Error generating embeddings for batch: {str(e)}[/]")
                    console.print(f"[red]Affected files: {current_files}[/]")
                    traceback.print_exc()
            
            return all_embeddings, all_documents, all_metadata
            
        except Exception as e:
            console.print(f"[bold red]Error in parallel processing: {str(e)}[/]")
            traceback.print_exc()
            raise
        finally:
            self.process_pool.shutdown(wait=True)
            self.thread_pool.shutdown(wait=True)
            
    async def process_distributed(self,
                                files: List[Path],
                                chunk_size: int = 20,
                                embedding_client: Any = None,
                                worker_addresses: List[str] = None,
                                progress: Optional[Progress] = None) -> Tuple[List, List, List]:
        """Process files using distributed workers
        
        Args:
            files: List of files to process
            chunk_size: Size of code chunks
            embedding_client: Client for generating embeddings
            worker_addresses: List of worker addresses to distribute work to
            progress: Optional progress tracker
            
        Returns:
            Tuple of (all_embeddings, all_documents, all_metadata)
        """
        # For now, just process locally using parallel processing
        return await self.process_files_parallel(
            files,
            chunk_size=chunk_size,
            embedding_client=embedding_client,
            progress=progress
        ) 