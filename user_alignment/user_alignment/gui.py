import tkinter as tk
from tkinter import scrolledtext
import threading
import queue

class UserAlignmentGUI:
    """
    Simple GUI for user alignment feedback.
    Runs in a separate thread from the ROS2 node.
    """
    def __init__(self):
        self.response_queue = queue.Queue()  # GUI → ROS: user feedback
        self.message_queue = queue.Queue()   # ROS → GUI: messages to display
        self.ready = threading.Event()
        
        # Start GUI in its own thread
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        
        # Wait for GUI to be ready
        self.ready.wait()

    def run(self):
        """Runs in a dedicated thread."""
        self.root = tk.Tk()
        self.root.title("User Alignment")
        self.root.geometry("500x400")

        # Display area for VLM responses
        tk.Label(self.root, text="Logs", font=("Arial", 11, "bold")).pack(pady=(10,0))
        self.display = scrolledtext.ScrolledText(self.root, height=10, state='disabled', wrap=tk.WORD)
        self.display.pack(fill=tk.X, padx=10, pady=5)

        # User input area
        tk.Label(self.root, text="Your feedback (or press OK if correct):").pack()
        self.entry = tk.Entry(self.root, width=50)
        self.entry.pack(padx=10, pady=5)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="OK", width=10, bg="green", fg="white",
                  command=self.on_ok).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Send correction", width=15,
                  command=self.on_send).pack(side=tk.LEFT, padx=5)

        # Object buttons frame (for handover — future)
        self.object_frame = tk.LabelFrame(self.root, text="Handover objects (future)")
        self.object_frame.pack(fill=tk.X, padx=10, pady=10)

        self.ready.set()

        # Poll for messages from ROS node
        self.root.after(100, self.poll_messages)
        self.root.mainloop()

    def poll_messages(self):
        """Check if ROS node sent a message to display."""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                self.append_text(msg)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_messages)

    def append_text(self, text):
        self.display.config(state='normal')
        self.display.insert(tk.END, text + "\n")
        self.display.see(tk.END)
        self.display.config(state='disabled')

    def on_ok(self):
        self.response_queue.put("ok")
        self.display_message("# User clicked 'ok'.")
        self.entry.delete(0, tk.END)

    def on_send(self):
        text = self.entry.get().strip()
        if text:
            self.response_queue.put(text)
            self.display_message(f"User: {text}")
            self.entry.delete(0, tk.END)

    # --- Public API called from ROS node ---

    def display_message(self, text):
        """Send a message to be displayed in the GUI (thread-safe)."""
        self.message_queue.put(text)

    def get_user_input(self, timeout=60.0):
        """
        Block until the user submits feedback or timeout.
        Returns 'ok' if user clicked OK, or the correction string.
        """
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return "ok"  # default to ok if user doesn't respond
