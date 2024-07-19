import tkinter as tk
from tkinter import ttk, filedialog
import os
from ..resources.animation import ProgressBar

class Functionality:
    def __init__(self):
        self.button_frame = ttk.Frame(self, padding=10)
        self.button_frame.pack(side="top", fill="x")

        self.file_path_button()
        self.create_button("Run Scripts", self.run_scripts)
        self.sys_button()

        self.message_box = tk.Text(self, height=10, state="disabled")
        self.message_box.pack(side="bottom", fill="both", expand=True)

        self.selected_process = tk.StringVar()
        self.create_radio_button("PDF Extraction", "pdf_extraction")
        self.create_radio_button("Text Formatting - Regex", "text_formatting")
        self.create_radio_button("Data Analysis", "data_analysis")

        self.progress_bar = ProgressBar()

    def file_path_button(self):
        file_path_frame = ttk.Frame(self.button_frame, padding=10)
        file_path_frame.pack(side="left")
        self.file_path_entry = ttk.Entry(file_path_frame, width=30)
        self.file_path_entry.pack(side="left", padx=5, pady=5)
        ttk.Button(file_path_frame, text="Browse Files", command=self.browse_file_path, style="Browse.TButton").pack(side="left", padx=5, pady=5)

    def browse_file_path(self):
        self.file_path = filedialog.askdirectory()
        self.file_path_entry.delete(0, tk.END)
        self.file_path_entry.insert(0, self.file_path)

    def sys_button(self):
        quit_button = ttk.Button(self.button_frame, text="QUIT", command=self.master.quit, style="Quit.TButton")
        quit_button.pack(side="right", padx=10, pady=10)

    def run_scripts(self):
        if not self.file_path:
            self.print_message("No path is given. Please provide a valid path!")
            return

        if not self.selected_process.get():
            self.print_message("No process is selected. Please select a process!")
            return

        files = os.listdir(self.file_path)
        txt_files = [file for file in files if file.endswith(".txt")]

        if not files:
            self.print_message("There is no file in the path. Please provide a valid path!")
            return
        elif not txt_files:
            self.print_message("There is no txt file in the path. Please provide a valid path!")
            return
        elif len(txt_files) != len(files):
            self.print_message("There should be only txt files inside the chosen document. Please provide a valid path!")
            return

        script_location = self.file_path
        self.print_message(f"Running scripts from {script_location}")

        self.progress_animation.start()

        if self.selected_process.get() == "pdf_extraction":
            # Perform PDF extraction process
            pass
        elif self.selected_process.get() == "text_formatting":
            # Perform text formatting process
            pass
        elif self.selected_process.get() == "data_analysis":
            # Perform data analysis process
            pass

        self.progress_animation.stop()

    def create_button(self, text, command):
        ttk.Button(self.button_frame, text=text, command=command, style="Run.TButton").pack(side="left", padx=5, pady=5)

    def create_radio_button(self, text, value):
        ttk.Radiobutton(self.button_frame, text=text, variable=self.selected_process, value=value).pack(side="left", padx=5, pady=5)

    def print_message(self, message):
        self.message_box.configure(state="normal")
        self.message_box.insert(tk.END, message + "\n")
        self.message_box.configure(state="disabled")
        self.message_box.see(tk.END)

def apply_styles(app):
    style = ttk.Style()
    style.configure("Browse.TButton", font=("Arial", 12), padding=6)
    style.configure("Quit.TButton", font=("Arial", 12), padding=6, foreground="red")
    style.configure("Run.TButton", font=("Arial", 12), padding=6, foreground="green")

    app.master.title("Ottoman NLP Toolkit")
    app.master.maxsize(1200, 700)
    app.master.minsize(600, 400)
    try:
        app.master.iconphoto(True, tk.PhotoImage(file=app.bg_image_path))
    except tk.TclError:
        print("Icon file not found.")
