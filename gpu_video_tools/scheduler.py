"""Async scheduler for batch job processing with per-device concurrency."""

import asyncio
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import Config
from .gpu import resolve_device, ResolvedDevice
from .bench import BenchmarkTimer, BenchmarkLogger


@dataclass
class BatchJob:
    """Represents a single job from batch CSV."""
    
    tool: str
    device_spec: Optional[str]
    args: Dict[str, str]
    row_number: int


def parse_batch_csv(csv_path: str) -> List[BatchJob]:
    """Parse batch CSV file into jobs.
    
    Expected CSV format:
        tool,device,input,output,<tool-specific columns>
    
    Args:
        csv_path: Path to batch CSV file
    
    Returns:
        List of BatchJob instances
    """
    jobs = []
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (after header)
            tool = row.get('tool', '').strip()
            if not tool:
                continue
            
            device_spec = row.get('device', '').strip() or None
            
            # Collect all other columns as args
            args = {k: v for k, v in row.items() if k not in ('tool', 'device')}
            
            jobs.append(BatchJob(
                tool=tool,
                device_spec=device_spec,
                args=args,
                row_number=row_num,
            ))
    
    return jobs


class JobScheduler:
    """Scheduler for running batch jobs with per-device concurrency limits."""
    
    def __init__(
        self,
        config: Config,
        policy: str = 'balance',
        bench_csv: Optional[str] = None,
    ):
        """Initialize job scheduler.
        
        Args:
            config: Configuration instance
            policy: Device resolution policy
            bench_csv: Optional path to benchmark CSV for logging
        """
        self.config = config
        self.policy = policy
        self.bench_logger = BenchmarkLogger(bench_csv) if bench_csv else None
        
        # Create semaphores for device concurrency limits
        self.device_semaphores: Dict[str, asyncio.Semaphore] = {}
        limits = config.get_device_limits()
        
        for device_spec, limit in limits.items():
            self.device_semaphores[device_spec] = asyncio.Semaphore(limit)
        
        # Default semaphore for devices not in config
        self.default_semaphore = asyncio.Semaphore(1)
    
    def get_semaphore(self, device: ResolvedDevice) -> asyncio.Semaphore:
        """Get the semaphore for a device.
        
        Args:
            device: Resolved device
        
        Returns:
            Semaphore for device concurrency control
        """
        return self.device_semaphores.get(device.device_spec, self.default_semaphore)
    
    async def run_job(self, job: BatchJob, console: Any) -> Dict[str, Any]:
        """Run a single job.
        
        Args:
            job: BatchJob to execute
            console: Console for output
        
        Returns:
            Dict with job result
        """
        timer = BenchmarkTimer()
        timer.start()
        
        try:
            # Resolve device
            device = self.config.get_device_for_tool(
                tool_name=job.tool,
                cli_device=None,
                batch_device=job.device_spec,
                policy=self.policy,
            )
            
            # Acquire device semaphore
            semaphore = self.get_semaphore(device)
            
            async with semaphore:
                # Execute job
                # NOTE: This is a simplified simulation. In a production system, 
                # this would call the actual tool functions (transcode, scenes, etc.)
                # with the appropriate arguments from job.args
                console.log(f"[Row {job.row_number}] Running {job.tool} on {device.device_spec}...")
                
                # Simulate work (in production, this would call actual tool functions)
                await asyncio.sleep(0.1)
                
                success = True
                notes = ""
        
        except Exception as e:
            success = False
            notes = str(e)
            console.log(f"[red][Row {job.row_number}] Failed: {e}[/red]")
        
        finally:
            timer.stop()
        
        # Log result
        if self.bench_logger:
            self.bench_logger.log_result(
                tool=job.tool,
                device=device if 'device' in locals() else None,
                timer=timer,
                success=success,
                input_path=job.args.get('input', ''),
                output_path=job.args.get('output', ''),
                notes=notes,
            )
        
        return {
            'row': job.row_number,
            'tool': job.tool,
            'device': device.device_spec if 'device' in locals() else 'unknown',
            'success': success,
            'duration': timer.duration,
            'notes': notes,
        }
    
    async def run_batch(self, jobs: List[BatchJob], console: Any) -> List[Dict[str, Any]]:
        """Run all batch jobs concurrently (within device limits).
        
        Args:
            jobs: List of BatchJob instances
            console: Console for output
        
        Returns:
            List of job results
        """
        console.log(f"Running {len(jobs)} jobs from batch...")
        
        # Create tasks
        tasks = [self.run_job(job, console) for job in jobs]
        
        # Run all tasks concurrently, collecting both successes and failures
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and convert exceptions to error dicts
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'row': jobs[i].row_number,
                    'tool': jobs[i].tool,
                    'device': 'unknown',
                    'success': False,
                    'duration': 0.0,
                    'notes': str(result),
                })
            else:
                processed_results.append(result)
        
        return processed_results


def run_batch_sync(
    csv_path: str,
    config: Config,
    policy: str = 'balance',
    bench_csv: Optional[str] = None,
    console: Any = None,
) -> List[Dict[str, Any]]:
    """Run batch jobs synchronously (wrapper for async function).
    
    Args:
        csv_path: Path to batch CSV file
        config: Configuration instance
        policy: Device resolution policy
        bench_csv: Optional benchmark CSV path
        console: Console for output
    
    Returns:
        List of job results
    """
    jobs = parse_batch_csv(csv_path)
    
    if not jobs:
        if console:
            console.log("[yellow]No jobs found in batch CSV[/yellow]")
        return []
    
    scheduler = JobScheduler(config, policy, bench_csv)
    
    # Run async event loop
    results = asyncio.run(scheduler.run_batch(jobs, console))
    
    return results
