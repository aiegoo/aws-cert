#!/usr/bin/env python3
"""
Kindle Screenshot & OCR Control App
Simple GUI to control the entire workflow
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
from pathlib import Path
import threading

class KindleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kindle Screenshot & OCR Control")
        self.root.geometry("600x700")
        
        # Paths
        self.raw_dir = r"C:\Users\hsyyu\Documents\kindle_raw"
        self.clean_dir = r"/mnt/c/Users/hsyyu/Documents/kindle_clean"
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title = tk.Label(self.root, text="Kindle Screenshot & OCR Control", 
                        font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(self.root, textvariable=self.status_var, 
                               fg="blue", font=("Arial", 10))
        status_label.pack(pady=5)
        
        # Frame 1: Capture Settings
        frame1 = ttk.LabelFrame(self.root, text="1. Capture Settings", padding=10)
        frame1.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame1, text="Number of pages:").grid(row=0, column=0, sticky="w")
        self.pages_var = tk.StringVar(value="560")
        tk.Entry(frame1, textvariable=self.pages_var, width=10).grid(row=0, column=1, sticky="w")
        
        tk.Label(frame1, text="Wait time (seconds):").grid(row=1, column=0, sticky="w")
        self.wait_var = tk.StringVar(value="2")
        tk.Entry(frame1, textvariable=self.wait_var, width=10).grid(row=1, column=1, sticky="w")
        
        tk.Button(frame1, text="Start Capture (PowerShell)", 
                 command=self.start_capture, bg="green", fg="white",
                 width=25).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Frame 2: Crop Settings
        frame2 = ttk.LabelFrame(self.root, text="2. Crop Settings", padding=10)
        frame2.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame2, text="Left crop (sidebar):").grid(row=0, column=0, sticky="w")
        self.left_var = tk.StringVar(value="300")
        tk.Entry(frame2, textvariable=self.left_var, width=10).grid(row=0, column=1, sticky="w")
        
        tk.Label(frame2, text="Bottom crop (cutoff):").grid(row=1, column=0, sticky="w")
        self.bottom_var = tk.StringVar(value="100")
        tk.Entry(frame2, textvariable=self.bottom_var, width=10).grid(row=1, column=1, sticky="w")
        
        tk.Button(frame2, text="Deduplicate & Crop", 
                 command=self.run_dedup, bg="orange", fg="white",
                 width=25).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Frame 3: OCR
        frame3 = ttk.LabelFrame(self.root, text="3. OCR Processing", padding=10)
        frame3.pack(fill="x", padx=10, pady=5)
        
        tk.Button(frame3, text="Run OCR on Clean Screenshots", 
                 command=self.run_ocr, bg="blue", fg="white",
                 width=25).pack(pady=5)
        
        # Frame 4: Flashcards
        frame4 = ttk.LabelFrame(self.root, text="4. Generate Flashcards", padding=10)
        frame4.pack(fill="x", padx=10, pady=5)
        
        tk.Button(frame4, text="Generate Flashcards from OCR Text", 
                 command=self.gen_flashcards, bg="purple", fg="white",
                 width=25).pack(pady=5)
        
        # Frame 5: Quick Actions
        frame5 = ttk.LabelFrame(self.root, text="Quick Actions", padding=10)
        frame5.pack(fill="x", padx=10, pady=5)
        
        tk.Button(frame5, text="Open Raw Screenshots Folder", 
                 command=self.open_raw).pack(fill="x", pady=2)
        tk.Button(frame5, text="Open Clean Screenshots Folder", 
                 command=self.open_clean).pack(fill="x", pady=2)
        tk.Button(frame5, text="View OCR Text", 
                 command=self.view_ocr).pack(fill="x", pady=2)
        tk.Button(frame5, text="Open Flashcards in Browser", 
                 command=self.open_flashcards).pack(fill="x", pady=2)
        
        # Log
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(log_frame, height=10, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(self.log_text)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)
        
    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        
    def start_capture(self):
        pages = self.pages_var.get()
        wait = self.wait_var.get()
        
        self.log(f"Starting PowerShell capture: {pages} pages, {wait}s wait")
        self.status_var.set("Capturing... (check PowerShell window)")
        
        # Launch PowerShell script
        cmd = f'powershell.exe -ExecutionPolicy Bypass -File .\\capture_raw.ps1 -MaxPages {pages} -WaitTime {wait}'
        self.log(f"Command: {cmd}")
        
        messagebox.showinfo("Capture Started", 
                          f"PowerShell window will open.\n"
                          f"Press F11 in Kindle first, then press ENTER in PowerShell.\n"
                          f"Capturing {pages} pages...")
        
    def run_dedup(self):
        left = self.left_var.get()
        bottom = self.bottom_var.get()
        
        self.status_var.set("Deduplicating and cropping...")
        self.log(f"Starting deduplication: left={left}, bottom={bottom}")
        
        def run():
            cmd = [
                "python3", "deduplicate_and_crop.py",
                "--input", "/mnt/c/Users/hsyyu/Documents/kindle_raw",
                "--output", "/mnt/c/Users/hsyyu/Documents/kindle_clean",
                "--left", left,
                "--bottom", bottom
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.log(result.stdout)
            if result.stderr:
                self.log(f"Errors: {result.stderr}")
            
            self.status_var.set("Deduplication complete!")
            self.root.after(0, lambda: messagebox.showinfo("Done", "Deduplication complete!"))
        
        threading.Thread(target=run, daemon=True).start()
        
    def run_ocr(self):
        self.status_var.set("Running OCR...")
        self.log("Starting OCR on clean screenshots...")
        
        def run():
            cmd = [
                "python3", "extract_text_split_pages.py",
                "--input-dir", "/mnt/c/Users/hsyyu/Documents/kindle_clean",
                "--output", "output/kindle_text_final.txt"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.log(result.stdout)
            if result.stderr:
                self.log(f"Errors: {result.stderr}")
            
            self.status_var.set("OCR complete!")
            self.root.after(0, lambda: messagebox.showinfo("Done", "OCR complete!"))
        
        threading.Thread(target=run, daemon=True).start()
        
    def gen_flashcards(self):
        self.status_var.set("Generating flashcards...")
        self.log("Creating flashcards from OCR text...")
        
        def run():
            cmd = [
                "python3", "generate_flashcards_from_text.py",
                "--input", "output/kindle_text_final.txt",
                "--output", "output/kindle_flashcards.html"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.log(result.stdout)
            if result.stderr:
                self.log(f"Errors: {result.stderr}")
            
            self.status_var.set("Flashcards ready!")
            self.root.after(0, lambda: messagebox.showinfo("Done", "Flashcards generated!"))
        
        threading.Thread(target=run, daemon=True).start()
        
    def open_raw(self):
        subprocess.run(['explorer', r'C:\Users\hsyyu\Documents\kindle_raw'])
        
    def open_clean(self):
        subprocess.run(['explorer', r'C:\Users\hsyyu\Documents\kindle_clean'])
        
    def view_ocr(self):
        subprocess.run(['notepad', 'output/kindle_text_final.txt'])
        
    def open_flashcards(self):
        path = Path("output/kindle_flashcards.html").absolute()
        subprocess.run(['start', str(path)], shell=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = KindleApp(root)
    root.mainloop()
