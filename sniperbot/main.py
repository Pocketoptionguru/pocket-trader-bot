import logging
import sys
import datetime
import tkinter as tk
from tkinter import messagebox

from bot import EnhancedMiniSwingTradingBot


def _setup_logging():
    log_file = f"enhanced_mini_swing_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logger initialised – output going to %s", log_file)


class MainApp(tk.Tk):
    """Very small Tkinter GUI that mimics the original functionality: a window
    with a single *Start Bot* button which kicks-off a background trading
    session.  All the visual bells & whistles from the original file have been
    removed to keep the module compact while still fulfilling the functional
    specification."""

    def __init__(self):
        super().__init__()
        self.title("Enhanced Mini Swing – Python Modular Split")
        self.geometry("400x200")
        self.bot = EnhancedMiniSwingTradingBot()
        self._create_widgets()

    def _create_widgets(self):
        self.start_btn = tk.Button(self, text="Start Bot", font=("Arial", 14, "bold"), command=self._start_bot)
        self.start_btn.pack(expand=True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _start_bot(self):
        self.start_btn.config(state='disabled')
        messagebox.showinfo("Bot", "Trading bot starting in background. Check logs for progress.")
        self.bot.run_trading_session()

    def _on_close(self):
        if messagebox.askokcancel("Quit", "Close application?"):
            self.destroy()


def main():
    _setup_logging()
    root = MainApp()
    root.mainloop()


if __name__ == '__main__':
    main()