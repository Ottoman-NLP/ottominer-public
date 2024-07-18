import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import os


class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack(side="left", fill="both", expand=True)
        self.file_path = ""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(script_dir, "items", "logo_llm.png")
        self.bg_image = logo_path
        # Load and set the background image with transparency
        try:
            self.bg_image = Image.open(self.bg_image)
            self.bg_image = self.bg_image.copy()
            self.bg_image.putalpha(128)  # Set transparency (0-255, where 0 is fully transparent)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
            self.bg_label = tk.Label(self, image=self.bg_photo)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            self.bg_label.lower()
        except FileNotFoundError:
            print("Background image file not found.")

        self.button_frame = ttk.Frame(self, padding=10)
        self.button_frame.pack(side="top", fill="x")

        self.file_path_button()
        self.create_button("Run Scripts", self.run_scripts)
        self.sys_button()

        self.message_box = tk.Text(self, height=10, state="disabled")
        self.message_box.pack(side="bottom", fill="both", expand=True)

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
            self.print_message("No path is given. Please provide a valid path.")
            return
        
        files = os.listdir(self.file_path)
        txt_files = [file for file in files if file.endswith(".txt")]
        
        if not files:
            self.print_message("There is no file in the path.")
            return
        elif not txt_files:
            self.print_message("There is no txt file in the path.")
            return
        elif len(txt_files) != len(files):
            self.print_message("There should be only txt files inside the chosen document.")
            return
        
        script_location = self.file_path
        self.print_message(f"Running scripts from {script_location}")

    def create_button(self, text, command):
        ttk.Button(self.button_frame, text=text, command=command, style="Run.TButton").pack(side="left", padx=5, pady=5)

    def print_message(self, message):
        self.message_box.configure(state="normal")
        self.message_box.insert(tk.END, message + "\n")
        self.message_box.configure(state="disabled")
        self.message_box.see(tk.END)

mapp = App()

# Configure styles for buttons
style = ttk.Style()
style.configure("Browse.TButton", font=("Arial", 12), padding=6)
style.configure("Quit.TButton", font=("Arial", 12), padding=6, foreground="red")
style.configure("Run.TButton", font=("Arial", 12), padding=6, foreground="green")

mapp.master.title("Ottoman NLP Toolkit")
mapp.master.maxsize(1200, 700)
mapp.master.minsize(600, 400)
try:
    mapp.master.iconphoto(True, tk.PhotoImage(file="items/logo.png"))
except tk.TclError:
    print("Icon file not found.")
mapp.mainloop()
