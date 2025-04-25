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
import psutil
import json
import os

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
        
        # Get file last modified time
        last_modified = Path(file_path).stat().st_mtime
        
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
                'has_comments': bool(re.search(r'#.*$', chunk_content, re.MULTILINE)),
                'last_modified': last_modified  # Add last modified timestamp
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
    def __init__(self, max_workers: Optional[int] = None, memory_limit_percentage: float = 80.0, 
                 cache_dir: str = '.cache', stream_to_disk: bool = True):
        """Initialize parallel processor with optional worker count and memory limits
        
        Args:
            max_workers: Maximum number of workers to use (defaults to CPU count)
            memory_limit_percentage: Maximum percentage of system memory to use before throttling
            cache_dir: Directory to store temporary results when streaming to disk
            stream_to_disk: Whether to stream results to disk instead of keeping in memory
        """
        self.memory_limit_percentage = memory_limit_percentage
        self.max_workers = self._calculate_optimal_workers(max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.cache_dir = cache_dir
        self.stream_to_disk = stream_to_disk
        
        # Create cache directory if it doesn't exist and streaming is enabled
        if self.stream_to_disk:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _calculate_optimal_workers(self, requested_workers: Optional[int] = None) -> int:
        """Calculate the optimal number of workers based on system resources"""
        # Get available CPU cores and system memory
        cpu_count = mp.cpu_count()
        available_memory = psutil.virtual_memory().available
        total_memory = psutil.virtual_memory().total
        
        # Start with requested workers or CPU count
        optimal_workers = requested_workers or cpu_count
        
        # If memory is tight, reduce worker count to avoid OOM
        memory_per_worker = 256 * 1024 * 1024  # Estimate 256MB per worker
        max_workers_by_memory = int(available_memory / memory_per_worker)
        
        # Use the more conservative estimate
        optimal_workers = min(optimal_workers, max_workers_by_memory)
        
        # Never go below 1 worker
        optimal_workers = max(1, optimal_workers)
        
        console.print(f"[dim]System has {cpu_count} CPUs, {available_memory/(1024*1024*1024):.1f}GB available memory[/]")
        console.print(f"[dim]Using {optimal_workers} workers for processing[/]")
        
        return optimal_workers

    def _get_adaptive_batch_size(self, default_batch_size: int = 10) -> int:
        """Calculate adaptive batch size based on available system memory"""
        mem = psutil.virtual_memory()
        
        # If memory usage is above threshold, reduce batch size
        if mem.percent > self.memory_limit_percentage:
            # Calculate how much we're exceeding our limit
            excess_factor = mem.percent / self.memory_limit_percentage
            # Reduce batch size proportionally, minimum 1
            return max(1, int(default_batch_size / excess_factor))
        
        return default_batch_size

    def _stream_results_to_disk(self, embeddings, documents, metadata, batch_index: int) -> str:
        """Save batch results to disk and return the file path"""
        batch_filename = f"{self.cache_dir}/batch_{batch_index}.json"
        with open(batch_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'embeddings': embeddings,
                'documents': documents,
                'metadata': metadata
            }, f)
        return batch_filename
    
    def _load_results_from_disk(self, batch_files: List[str]) -> Tuple[List, List, List]:
        """Load and combine all batch results from disk"""
        all_embeddings = []
        all_documents = []
        all_metadata = []
        
        for batch_file in batch_files:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                all_embeddings.extend(batch_data['embeddings'])
                all_documents.extend(batch_data['documents'])
                all_metadata.extend(batch_data['metadata'])
                
        return all_embeddings, all_documents, all_metadata
        
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
            
            # Split files into manageable chunks to avoid memory issues
            max_files_per_batch = min(100, max(10, int(len(files) / self.max_workers)))
            
            # Track if we need to clean up temporary files
            temp_batch_files = []
            
            # Lists to store results if not streaming to disk
            all_embeddings = []
            all_documents = []
            all_metadata = []
            
            # Process files in batches to manage memory
            for file_batch_start in range(0, len(files), max_files_per_batch):
                file_batch = files[file_batch_start:file_batch_start + max_files_per_batch]
                
                # Submit file processing tasks
                chunk_futures = []
                for file_path in file_batch:
                    future = loop.run_in_executor(
                        self.process_pool,
                        process_file,
                        str(file_path),
                        chunk_size
                    )
                    chunk_futures.append(future)
                
                # Collect processing results
                results = []
                for i, future in enumerate(asyncio.as_completed(chunk_futures)):
                    try:
                        current_file = str(file_batch[i])
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
                        console.print(f"[red]Error processing {file_batch[i]}: {str(e)}[/]")
                        traceback.print_exc()
                
                # Generate embeddings for the processing results
                if embedding_client:
                    # Process in smaller batches based on system resources
                    for embed_batch_start in range(0, len(results), batch_size):
                        # Check memory and adjust batch size dynamically
                        adaptive_batch_size = self._get_adaptive_batch_size(batch_size)
                        if adaptive_batch_size < batch_size:
                            console.print(f"[yellow]Memory usage high, reducing batch size from {batch_size} to {adaptive_batch_size}[/]")
                        
                        # Get batch of results
                        batch = results[embed_batch_start:embed_batch_start + adaptive_batch_size]
                        
                        # Collect all texts from this batch
                        batch_texts = []
                        batch_files = []
                        for result in batch:
                            batch_files.append(Path(result.file_path).name)
                            for chunk in result.chunks:
                                batch_texts.append(chunk['content'])
                        
                        # Skip if no texts to embed in this batch
                        if not batch_texts:
                            continue
                        
                        current_files = ", ".join(batch_files[:3])
                        if len(batch_files) > 3:
                            current_files += f" (+{len(batch_files)-3} more)"
                        
                        if progress and task_embed:
                            progress.update(task_embed, description=f"[yellow]Generating embeddings for: {current_files}")
                        
                        # Generate embeddings for the batch
                        try:
                            console.print(f"[dim]Embedding batch of {len(batch_texts)} chunks from {len(batch_files)} files[/]")
                            batch_embeddings = await embedding_client.get_embeddings(batch_texts)
                            
                            # Prepare batch data
                            batch_documents = []
                            batch_metadata = []
                            
                            # Store results
                            text_idx = 0
                            for result in batch:
                                for chunk in result.chunks:
                                    if text_idx < len(batch_embeddings):
                                        batch_documents.append(chunk['content'])
                                        batch_metadata.append(chunk['metadata'])
                                    text_idx += 1
                            
                            # Either store in memory or stream to disk
                            if self.stream_to_disk:
                                batch_file = self._stream_results_to_disk(
                                    batch_embeddings, batch_documents, batch_metadata, 
                                    len(temp_batch_files)
                                )
                                temp_batch_files.append(batch_file)
                                # Free up memory
                                del batch_embeddings
                                del batch_documents
                                del batch_metadata
                            else:
                                # Add to in-memory lists
                                all_embeddings.extend(batch_embeddings)
                                all_documents.extend(batch_documents)
                                all_metadata.extend(batch_metadata)
                                
                            if progress and task_embed:
                                progress.update(task_embed, advance=len(batch))
                                
                        except Exception as e:
                            console.print(f"[red]Error generating embeddings for batch: {str(e)}[/]")
                            console.print(f"[red]Affected files: {current_files}[/]")
                            traceback.print_exc()
            
            # Combine results from disk if streaming was used
            if self.stream_to_disk and temp_batch_files:
                console.print(f"[dim]Combining {len(temp_batch_files)} result batches from disk[/]")
                all_embeddings, all_documents, all_metadata = self._load_results_from_disk(temp_batch_files)
                
                # Clean up temporary files
                for batch_file in temp_batch_files:
                    try:
                        os.remove(batch_file)
                    except Exception:
                        pass
            
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