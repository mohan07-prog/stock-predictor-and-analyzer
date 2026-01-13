import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import io
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinterdnd2 import TkinterDnD, DND_FILES


# --- (1) USER SETTINGS: Please update these ---

# Set the names of the columns you want to analyze
# (Case-sensitive!)
DATE_COLUMN = "Date"
VALUE_COLUMN = "Close"

# Set the period for the Simple Moving Average
SMA_PERIOD = 20

# -----------------------------------------------

def analyze_data(file_content, date_col, value_col, sma_period, file_name="uploaded_file.csv"):
    """
    Loads data from file content, calculates statistics, SMA, and a trend line,
    then generates a plot.
    """
    try:
        # --- (2) Load and Prepare Data ---
        print(f"Loading data from '{file_name}'...")
        df = pd.read_csv(io.StringIO(file_content))
    except Exception as e:
        print(f"An error occurred loading the CSV: {e}")
        return

    # --- (3) Validate Columns ---
    if date_col not in df.columns or value_col not in df.columns:
        print(f"--- ERROR ---")
        print(f"Columns not found. Script expected '{date_col}' and '{value_col}'.")
        print(f"Available columns are: {list(df.columns)}")
        print("Please update the 'DATE_COLUMN' and 'VALUE_COLUMN' variables.")
        print(f"-----------------")
        return

    try:
        # --- (4) Process Data ---
        print("Processing data...")
        # Ensure date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # Sort by date to ensure correct SMA calculation
        df = df.sort_values(by=date_col)

        # Ensure value column is numeric
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

        # Drop any rows where conversion failed
        df = df.dropna(subset=[value_col])

        if df.empty:
            print("No valid numeric data found in the selected value column.")
            return

        values = df[value_col]

        # --- (5) Calculate Statistics ---
        mean_val = values.mean()
        min_val = values.min()
        max_val = values.max()

        print("\n--- Analysis Results ---")
        print(f"Mean Value:    {mean_val:,.2f}")
        print(f"Minimum Value: {min_val:,.2f}")
        print(f"Maximum Value: {max_val:,.2f}")
        print("------------------------\n")

        # --- (6) Calculate Indicators ---

        # Simple Moving Average (SMA)
        df['SMA'] = values.rolling(window=sma_period).mean()

        # Predicted Trend (Linear Regression)
        # We need a numeric version of the date for regression
        df['date_numeric'] = df[date_col].map(pd.Timestamp.toordinal)
        X = df['date_numeric'].values
        y = values.values

        # Fit a 1st-degree polynomial (a straight line)
        # m = slope, b = intercept
        m, b = np.polyfit(X, y, 1)

        # Create the trendline values
        df['Trend'] = (m * X) + b

        # --- (7) Plot Data ---
        print("Generating plot...")
        plt.figure(figsize=(14, 7))

        # Main data plot
        plt.plot(df[date_col], df[value_col], label=value_col, color='blue', alpha=0.8)

        # SMA plot
        plt.plot(df[date_col], df['SMA'], label=f'{sma_period}-Period SMA', color='red', linestyle='--')

        # Trendline plot
        plt.plot(df[date_col], df['Trend'], label='Predicted Trend', color='purple', linestyle='--')

        # Formatting
        plt.title(f"Data Analysis for '{file_name}'")
        plt.xlabel(date_col)
        plt.ylabel(value_col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Show the plot
        print("Displaying plot. Close the plot window to exit.")
        plt.show()

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        print("Please check your column names and data types.")

# Create a drag and drop window
class DragDropWindow:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Stock Data Analyzer - Drag and Drop CSV")
        self.root.geometry("600x300")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create and configure drop zone
        self.drop_zone = ttk.Frame(self.main_frame, padding="30")
        self.drop_zone.pack(fill=tk.BOTH, expand=True)
        
        # Add labels
        self.label = ttk.Label(self.drop_zone, 
                             text="Drag and Drop your CSV file here\nor click to select a file",
                             font=('Arial', 12))
        self.label.pack(pady=20)
        
        # Status label
        self.status_label = ttk.Label(self.main_frame, text="")
        self.status_label.pack(pady=10)
        
        # Bind click event
        self.drop_zone.bind('<Button-1>', self.handle_click)
        
        # Enable drop zone to receive files
        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind('<<Drop>>', self.handle_drop)
        
        # Style the drop zone
        self.drop_zone.configure(relief="groove", borderwidth=2)
        
    def handle_drop(self, event):
        file_path = event.data
        # Remove curly braces that TkinterDnD adds
        file_path = file_path.strip('{}')
        if file_path.lower().endswith('.csv'):
            self.process_file(file_path)
        else:
            messagebox.showerror("Error", "Please drop a CSV file")
            
    def handle_click(self, event):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")]
        )
        if file_path:
            self.process_file(file_path)
            
    def process_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                file_content = file.read()
                self.status_label.config(text=f"Processing {os.path.basename(file_path)}...")
                self.root.update()
                analyze_data(file_content, DATE_COLUMN, VALUE_COLUMN, SMA_PERIOD, os.path.basename(file_path))
                self.status_label.config(text="Analysis complete!")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error processing file: {str(e)}")
    
    def run(self):
        # Initialize tkinter drag and drop support
        self.setup_tkdnd()
        self.root.mainloop()
        
    def setup_tkdnd(self):
        # Nothing to setup for tkinterdnd2
        pass

# --- Main execution ---
if __name__ == "__main__":
    try:
        app = DragDropWindow()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("If you see a tkdnd error, please install it using: pip install tkdnd")