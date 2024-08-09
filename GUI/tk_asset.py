from tkinter import Tk
from PIL import Image, ImageTk

def set_app_icon(root: Tk, icon_path: str):
    try:
        ico = Image.open(icon_path)
        photo = ImageTk.PhotoImage(ico)
        root.wm_iconphoto(True, photo)
    except FileNotFoundError:
        print(f"Icon file not found at {icon_path}.")
