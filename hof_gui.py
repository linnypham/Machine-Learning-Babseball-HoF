import tkinter as tk
from hof import predict_player_hof 

# --- Modify predict_player_hof to return a result string ---
def predict_player_hof_gui(first, last):
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    try:
        predict_player_hof(first, last)
    except Exception as e:
        print(f"Error: {e}")
    sys.stdout = old_stdout
    return mystdout.getvalue().strip()

# --- Tkinter GUI ---
def run_prediction():
    first = entry_first.get()
    last = entry_last.get()
    result = predict_player_hof_gui(first, last)
    output_label.config(text=result if result else "No result.")

root = tk.Tk()
root.title("ML Baseball Hall of Fame Predictor")
root.geometry("500x300")

tk.Label(root, text="First Name:").pack(pady=(20, 0))
entry_first = tk.Entry(root)
entry_first.pack()

tk.Label(root, text="Last Name:").pack(pady=(10, 0))
entry_last = tk.Entry(root)
entry_last.pack()

tk.Button(root, text="Predict HOF", command=run_prediction).pack(pady=20)

output_label = tk.Label(root, text="", font=("Helvetica", 12), justify="left", wraplength=480)
output_label.pack(pady=10)

root.mainloop()
