from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from datetime import datetime
import time
import threading
from typing import Optional, Dict, Any

class ProgressTracker:
    """Enhanced terminal UI progress tracking for Ottoman Miner"""
    
    def __init__(self):
        self.console = Console()
        self.done = False
        self.current_operation = ""
        self.current_file = ""
        self.total_files = 0
        self.processed_files = 0
        self.start_time = None
        self.stats = {}
        self._thread = None
        
        # Initialize progress bars
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
            expand=True
        )
        
        self.file_progress = Progress(
            TextColumn("[bold yellow]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("{task.completed}/{task.total}"),
            expand=True
        )

    def _create_layout(self) -> Layout:
        """Create the terminal UI layout"""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="progress", ratio=2),
            Layout(name="stats", ratio=1)
        )
        
        return layout

    def _generate_stats_table(self) -> Table:
        """Generate statistics table"""
        table = Table(show_header=False, expand=True)
        table.add_column("Metric")
        table.add_column("Value")
        
        for key, value in self.stats.items():
            table.add_row(f"[cyan]{key}[/cyan]", f"[yellow]{value}[/yellow]")
        
        return table

    def _update_display(self, live: Live, layout: Layout, task_id: int, file_task_id: int):
        """Update the terminal UI display"""
        while not self.done:
            # Update header
            layout["header"].update(
                Panel(
                    f"[bold white]Ottoman Miner - {self.current_operation}[/bold white]",
                    style="blue"
                )
            )
            
            # Update progress section
            progress_layout = Layout()
            progress_layout.update(
                Panel(
                    self.progress,
                    title="Overall Progress",
                    border_style="blue"
                )
            )
            
            file_layout = Layout()
            file_layout.update(
                Panel(
                    self.file_progress,
                    title="Current File",
                    border_style="yellow"
                )
            )
            
            layout["progress"].update(progress_layout)
            
            # Update stats section
            layout["stats"].update(
                Panel(
                    self._generate_stats_table(),
                    title="Statistics",
                    border_style="cyan"
                )
            )
            
            # Update footer
            elapsed = time.time() - self.start_time if self.start_time else 0
            layout["footer"].update(
                Panel(
                    f"Elapsed: {elapsed:.1f}s | Memory: {self.stats.get('Memory', 'N/A')}",
                    style="blue"
                )
            )
            
            # Update progress
            self.progress.update(task_id, completed=self.processed_files)
            if self.current_file:
                self.file_progress.update(file_task_id, description=self.current_file)
            
            time.sleep(0.1)

    def start(self, operation: str, total_files: int):
        """Start tracking progress with terminal UI"""
        self.done = False
        self.current_operation = operation
        self.total_files = total_files
        self.processed_files = 0
        self.start_time = time.time()
        self.stats = {}
        
        layout = self._create_layout()
        
        # Create main progress task
        task_id = self.progress.add_task(
            operation,
            total=total_files
        )
        
        # Create file progress task
        file_task_id = self.file_progress.add_task(
            "Current file",
            total=100
        )
        
        # Start live display
        with Live(layout, refresh_per_second=10, screen=True):
            self._thread = threading.Thread(
                target=self._update_display,
                args=(Live, layout, task_id, file_task_id)
            )
            self._thread.daemon = True
            self._thread.start()
    
    def update(self, file_name: str, stats: Optional[Dict[str, Any]] = None):
        """Update progress with current file and statistics"""
        self.current_file = file_name
        self.processed_files += 1
        if stats:
            self.stats = stats
    
    def stop(self):
        """Stop progress tracking"""
        self.done = True
        if self._thread:
            self._thread.join()
            self._thread = None
        self.console.print("\n[bold green]Operation completed successfully![/bold green]")

# Global instance
progress = ProgressTracker()

# Usage example
if __name__ == "__main__":
    # Simulate PDF processing
    files = [f"document_{i}.pdf" for i in range(5)]
    
    progress.start("PDF Extraction", len(files))
    
    for file in files:
        time.sleep(2)  # Simulate processing
        progress.update(file, {
            "Pages Processed": "10/10",
            "Text Extraction": "100%",
            "Memory Usage": "124MB",
            "OCR Quality": "High",
            "Errors": "None",
            "Processing Speed": "2.1 pages/s"
        })
    
    progress.stop()