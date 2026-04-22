"""
Parallel Executor - Multiprocessing experiment runner

Executes multiple experiments in parallel using ProcessPoolExecutor.
Fault-tolerant with safe result collection.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional
import traceback
import multiprocessing as mp


class ParallelExecutor:
    """
    Parallel experiment executor using multiprocessing.
    
    Features:
    - Configurable number of workers
    - Safe result collection
    - Fault-tolerant execution
    - Progress tracking
    - Clear logging
    
    Example:
        >>> executor = ParallelExecutor(max_workers=4)
        >>> tasks = [{"model": "RF", "dataset": "fake_news"}, ...]
        >>> results = executor.run(train_model, tasks)
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of parallel workers.
                        If None, uses CPU count.
        """
        if max_workers is None:
            max_workers = max(1, mp.cpu_count() - 1)
        
        self.max_workers = max_workers
        print(f"ParallelExecutor initialized with {self.max_workers} workers")
    
    def run(
        self,
        func: Callable,
        tasks: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute tasks in parallel.
        
        Args:
            func: Function to execute for each task.
                  Must accept a single dict argument and return a dict.
            tasks: List of task configurations (dicts)
            show_progress: Whether to show progress updates
            
        Returns:
            List of results (dicts) in same order as tasks
        """
        if not tasks:
            print("Warning: No tasks to execute")
            return []
        
        print(f"\nExecuting {len(tasks)} tasks with {self.max_workers} workers...")
        start_time = time.time()
        
        results = [None] * len(tasks)
        completed = 0
        failed = 0
        
        # Create task index mapping
        task_map = {i: task for i, task in enumerate(tasks)}
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._safe_execute, func, task, idx): idx
                for idx, task in task_map.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                
                try:
                    result = future.result()
                    results[idx] = result
                    
                    if result.get('status') == 'success':
                        completed += 1
                    else:
                        failed += 1
                    
                    if show_progress:
                        progress = (completed + failed) / len(tasks) * 100
                        print(f"Progress: {completed + failed}/{len(tasks)} "
                              f"({progress:.1f}%) - "
                              f"Success: {completed}, Failed: {failed}")
                
                except Exception as e:
                    print(f"Error collecting result for task {idx}: {e}")
                    results[idx] = {
                        'status': 'error',
                        'error': str(e),
                        'task_idx': idx
                    }
                    failed += 1
        
        duration = time.time() - start_time
        
        print(f"\nExecution complete!")
        print(f"Total time: {duration:.2f}s")
        print(f"Successful: {completed}/{len(tasks)}")
        print(f"Failed: {failed}/{len(tasks)}")
        print(f"Average time per task: {duration/len(tasks):.2f}s")
        
        return results
    
    @staticmethod
    def _safe_execute(func: Callable, task: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """
        Safely execute a task with error handling.
        
        Args:
            func: Function to execute
            task: Task configuration
            idx: Task index
            
        Returns:
            Result dictionary with status
        """
        try:
            start_time = time.time()

            # Execute task
            result = func(task)

            duration = time.time() - start_time

            # Ensure result is a dict for uniform handling
            if not isinstance(result, dict):
                result = {'result': result}

            # Preserve explicit failure status if caller already marked failure
            inner_status = result.get('status')
            if inner_status in ('failed', 'error'):
                result.setdefault('task_idx', idx)
                result.setdefault('duration', duration)
                result.setdefault('task_config', task)
                return result

            # Mark all valid task outputs as success
            result['status'] = 'success'
            result['task_idx'] = idx
            result['duration'] = duration

            return result

        except Exception as e:
            # Capture full traceback
            error_trace = traceback.format_exc()

            return {
                'status': 'failed',
                'task_idx': idx,
                'error': str(e),
                'traceback': error_trace,
                'task_config': task
            }
    
    def run_sequential(
        self,
        func: Callable,
        tasks: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute tasks sequentially (for debugging).
        
        Args:
            func: Function to execute
            tasks: List of task configurations
            show_progress: Whether to show progress
            
        Returns:
            List of results
        """
        print(f"\nExecuting {len(tasks)} tasks sequentially...")
        start_time = time.time()
        
        results = []
        
        for idx, task in enumerate(tasks):
            if show_progress:
                print(f"Task {idx+1}/{len(tasks)}")
            
            result = self._safe_execute(func, task, idx)
            results.append(result)
        
        duration = time.time() - start_time
        
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = len(results) - successful
        
        print(f"\nExecution complete!")
        print(f"Total time: {duration:.2f}s")
        print(f"Successful: {successful}/{len(tasks)}")
        print(f"Failed: {failed}/{len(tasks)}")
        
        return results


class BatchExecutor:
    """
    Execute tasks in batches to control memory usage.
    
    Useful when tasks are memory-intensive.
    """
    
    def __init__(self, max_workers: int = 4, batch_size: int = 10):
        """
        Initialize batch executor.
        
        Args:
            max_workers: Workers per batch
            batch_size: Number of tasks per batch
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ParallelExecutor(max_workers=max_workers)
    
    def run(
        self,
        func: Callable,
        tasks: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute tasks in batches.
        
        Args:
            func: Function to execute
            tasks: List of tasks
            show_progress: Show progress
            
        Returns:
            List of results
        """
        print(f"Executing {len(tasks)} tasks in batches of {self.batch_size}")
        
        all_results = []
        
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(tasks) + self.batch_size - 1) // self.batch_size
            
            print(f"\n=== Batch {batch_num}/{total_batches} ===")
            
            batch_results = self.executor.run(func, batch, show_progress)
            all_results.extend(batch_results)
        
        return all_results
