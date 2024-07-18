import tk_frame as tkframe
import tk_function as tkfunc
import meta as meta
import tkinter as tk
import tk_asset


class App(tkframe.App, tkfunc.Functionality):
    def __init__(self, master=None):
        tkframe.App.__init__(self, master)
        tkfunc.Functionality.__init__(self)
        self.metadata = meta.main()
        self.set_icon()

    def set_icon(self):
        tk_asset.set_app_icon(self.master, 'GUI/assets/logo.png')
    
    def run(self):
        self.master.mainloop()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = App(master=root)
        app.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

