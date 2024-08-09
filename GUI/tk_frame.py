import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

class App(tk.Frame):
    """
    This class is the main application frame. It contains the GUI elements for frame design.
    """
    def __init__(self, master=None):
        super().__init__(master)
        self.pack(side="left", fill="both", expand=True)
        self.file_path = ""

        try:
            self.bg_image = Image.open('GUI/assets/logo_llm.png')
            self.bg_image = self.bg_image.copy()
            self.bg_image.putalpha(200)  # Set transparency (0-255, where 0 is fully transparent)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
            self.bg_label = tk.Label(self, image=self.bg_photo)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            self.bg_label.lower()
        except FileNotFoundError:
            print("Background image file not found.")
