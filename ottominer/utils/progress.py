from rich.progress import Progress, SpinnerColumn, Task
from rich.console import Console
from rich.table import Table
from contextlib import contextmanager
from typing import Dict, Optional
import logging
import time

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Singleton progress tracker to avoid multiple live displays."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._progress = None
            cls._instance._console = Console()
            cls._instance._task_history = {}
            cls._instance._active = False
        return cls._instance

    def force_stop(self):
        """Force stop any active progress display."""
        if self._progress and self._active:
            try:
                # Store task info before stopping
                for task_id in self._progress.task_ids:
                    task = self._progress.tasks[task_id]
                    self._task_history[task_id] = {
                        'description': task.description,
                        'completed': task.completed,
                        'total': task.total
                    }
                self._progress.stop()
            except Exception as e:
                logger.warning(f"Error stopping progress: {e}")
            finally:
                self._progress = None
                self._active = False

    def __enter__(self):
        """Support context manager protocol."""
        self.force_stop()  # Stop any existing progress
        time.sleep(0.1)  # Give time for cleanup
        
        self._progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            console=self._console,
            transient=True
        )
        self._progress.start()
        self._active = True
        return self._progress

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit."""
        self.force_stop()

    def create_status_table(self) -> Table:
        """Create a rich table showing task status."""
        table = Table(title="Task Status")
        table.add_column("Task ID")
        table.add_column("Description")
        table.add_column("Progress")
        table.add_column("Status")

        for task_id, info in self._task_history.items():
            progress = f"{info['completed']}/{info['total']}"
            status = "Complete" if info['completed'] >= info['total'] else "In Progress"
            table.add_row(
                str(task_id),
                info['description'],
                progress,
                status
            )

        return table

    def get_task(self, task_id: int) -> Optional[Task]:
        """Get task by ID safely."""
        if self._progress and task_id in self._progress.tasks:
            return self._progress.tasks[task_id]
        return None

    def update_task(self, task_id: int, advance: int = 1):
        """Update task progress safely."""
        if self._progress:
            self._progress.update(task_id, advance=advance)