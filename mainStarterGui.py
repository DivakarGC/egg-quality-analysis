import tkinter as tk
import subprocess

def run_command():
    command = "streamlit run main.py" 
    #edit the above command with whatever you get when you run main.py
    subprocess.run(command, shell=True)

def main():
    root = tk.Tk()
    root.title("Egg Quality Analysis")
    root.geometry("700x330")
    button = tk.Button(root, text="LAUNCH APP", command=run_command, height=50, width=50)
    button.pack(pady=100)
    button_font = ("Helvetica", 50, "bold")
    root.minsize(700, 330)
    root.config(background="#333333")
    root.mainloop()
    

if __name__ == "__main__":
    main()
