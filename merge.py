import pandas as pd
import matplotlib
matplotlib.use("TkAgg") # Use the Tkinter backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import os

# --- UI Styling Constants ---
BG_COLOR = "#f0f2f5"
SIDEBAR_COLOR = "#1f2937"
CARD_COLOR = "#ffffff"
PRIMARY_COLOR = "#3b82f6"
PRIMARY_ACTIVE_COLOR = "#2563eb"
TEXT_COLOR = "#f9fafb"
LABEL_COLOR = "#374151"
SUCCESS_COLOR = "#10b981"
FONT_FAMILY = "Segoe UI"

class StressAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stress Fusion AI Analyzer")
        self.root.geometry("1200x950")
        self.root.configure(bg=BG_COLOR)
        
        self.survey_filepath = ""
        self.wearable_filepath = ""
        self.plot_canvases = []

        # --- Style Configuration ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=BG_COLOR)
        style.configure("Card.TFrame", background=CARD_COLOR, relief="raised", borderwidth=1)
        style.configure("TLabel", background=BG_COLOR, foreground=LABEL_COLOR, font=(FONT_FAMILY, 11))
        style.configure("Card.TLabel", background=CARD_COLOR, foreground=LABEL_COLOR, font=(FONT_FAMILY, 11))
        style.configure("Header.TLabel", font=(FONT_FAMILY, 24, "bold"))
        style.configure("SubHeader.TLabel", font=(FONT_FAMILY, 12), foreground="#4b5563")
        style.configure("Vertical.TScrollbar", background=PRIMARY_COLOR, troughcolor=BG_COLOR)


        self._create_widgets()

    def _create_widgets(self):
        # --- Main Layout (Sidebar + Content) ---
        sidebar = tk.Frame(self.root, bg=SIDEBAR_COLOR, width=280)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # --- Main Content Area ---
        main_canvas = tk.Canvas(self.root, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        self.scrollable_frame = ttk.Frame(main_canvas, style="TFrame")

        self.scrollable_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Populate Sidebar ---
        self._populate_sidebar(sidebar)
        
        # --- Results Frame (in scrollable area) ---
        self.results_frame = ttk.Frame(self.scrollable_frame, style="TFrame", padding="20 40 40 40")
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        self.results_frame.columnconfigure(0, weight=1)
        self._show_initial_message()

    def _populate_sidebar(self, sidebar):
        # --- Header ---
        tk.Label(sidebar, text="Stress Fusion", bg=SIDEBAR_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, 22, "bold")).pack(pady=20, padx=10)
        tk.Label(sidebar, text="AI Analyzer", bg=SIDEBAR_COLOR, fg="#9ca3af", font=(FONT_FAMILY, 14)).pack(pady=(0, 20), padx=10)

        # --- File Upload Section ---
        upload_frame = tk.Frame(sidebar, bg=SIDEBAR_COLOR)
        upload_frame.pack(fill="x", padx=20, pady=10)
        tk.Label(upload_frame, text="DATA SOURCES", bg=SIDEBAR_COLOR, fg="#9ca3af", font=(FONT_FAMILY, 10, "bold")).pack(fill="x", anchor="w")
        
        self.survey_label = self._create_file_selector(upload_frame, "Survey Data", self._select_survey_file)
        self.wearable_label = self._create_file_selector(upload_frame, "Wearable Data", self._select_wearable_file)

        # --- Spacer ---
        tk.Frame(sidebar, bg=SIDEBAR_COLOR).pack(pady=10)

        # --- Analysis Button ---
        self.analyze_button = tk.Button(sidebar, text="Analyze Data", font=(FONT_FAMILY, 12, "bold"),
                                        bg=PRIMARY_COLOR, fg=TEXT_COLOR, relief=tk.FLAT, borderwidth=0,
                                        activebackground=PRIMARY_ACTIVE_COLOR, activeforeground=TEXT_COLOR,
                                        state=tk.DISABLED, command=self._start_analysis_thread, cursor="hand2",
                                        anchor="center", justify="center", padx=20)
        self.analyze_button.pack(fill="x", padx=20, ipady=10)

        # --- Status Label ---
        self.status_label = tk.Label(sidebar, text="Please upload both files.", bg=SIDEBAR_COLOR, fg="#9ca3af", font=(FONT_FAMILY, 10), wraplength=240)
        self.status_label.pack(fill="x", padx=20, pady=20, side="bottom")

    def _create_file_selector(self, parent, text, command):
        frame = tk.Frame(parent, bg=SIDEBAR_COLOR)
        frame.pack(fill="x", pady=5)
        
        label = tk.Label(frame, text=text, font=(FONT_FAMILY, 11), bg=SIDEBAR_COLOR, fg=TEXT_COLOR, anchor="w", cursor="hand2")
        label.pack(side="left", fill="x", expand=True, padx=10)
        label.bind("<Button-1>", lambda e: command())
        
        return label
        
    def _select_survey_file(self):
        filepath = filedialog.askopenfilename(title="Select Survey CSV", filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.survey_filepath = filepath
            self.survey_label.config(text=f"Survey: Loaded", foreground=SUCCESS_COLOR)
            self._check_files_ready()

    def _select_wearable_file(self):
        filepath = filedialog.askopenfilename(title="Select Wearable CSV", filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.wearable_filepath = filepath
            self.wearable_label.config(text=f"Wearable: Loaded", foreground=SUCCESS_COLOR)
            self._check_files_ready()
            
    def _check_files_ready(self):
        if self.survey_filepath and self.wearable_filepath:
            self.analyze_button.config(state=tk.NORMAL)
            self.status_label.config(text="Ready to analyze!")

    def _show_initial_message(self):
        tk.Label(self.results_frame, text="Awaiting Data Analysis", bg=BG_COLOR, fg="#6b7280", font=(FONT_FAMILY, 28, "bold")).pack(pady=50)
        tk.Label(self.results_frame, text="Upload your datasets using the panel on the left and click 'Analyze Data' to generate the report.", bg=BG_COLOR, fg="#6b7280", font=(FONT_FAMILY, 14), wraplength=600).pack()

    def _start_analysis_thread(self):
        self.analyze_button.config(state=tk.DISABLED, text="Analyzing...")
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.plot_canvases.clear()
        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()
    
    def _run_analysis(self):
        try:
            self._update_status("Initializing analysis...", delay=0.2)
            survey_df = pd.read_csv(self.survey_filepath)
            wearable_df = pd.read_csv(self.wearable_filepath)
            
            self._update_status("Aligning datasets...")
            if 'Timestamp' in survey_df.columns: survey_df = survey_df.drop(columns=['Timestamp'])
            if 'Timestamp' in wearable_df.columns: wearable_df = wearable_df.drop(columns=['Timestamp'])
            n_survey, n_wearable = len(survey_df), len(wearable_df)
            if n_survey == 0 or n_wearable == 0: raise ValueError("One of the uploaded files is empty.")

            if n_wearable > n_survey:
                self._update_status("Aggregating high-frequency wearable data...")
                chunk_size = n_wearable // n_survey
                grouping_key = np.arange(n_wearable) // chunk_size
                wearable_df = wearable_df.iloc[:len(grouping_key)]
                wearable_df['group'] = grouping_key
                wearable_df = wearable_df[wearable_df['group'] < n_survey]
                aggregation_rules = {'EDA':'mean', 'TEMP':'mean', 'EMG':'mean', 'RESP':'mean', 'ECG':'mean', 'Predicted Stress':lambda x: x.mode()[0] if not x.empty else None}
                valid_rules = {k: v for k, v in aggregation_rules.items() if self._find_column(wearable_df, [k])}
                wearable_agg_df = wearable_df.groupby('group').agg(valid_rules)
                merged_df = pd.concat([survey_df.reset_index(drop=True), wearable_agg_df.reset_index(drop=True)], axis=1)
            else:
                self._update_status("Warning: Wearable data has fewer rows. Truncating.")
                min_rows = min(n_survey, n_wearable)
                merged_df = pd.concat([survey_df.iloc[:min_rows].reset_index(drop=True), wearable_df.iloc[:min_rows].reset_index(drop=True)], axis=1)

            if merged_df.empty: raise ValueError("Could not merge files.")
            self._update_status(f"Merged {len(merged_df)} records. Generating report...")

            survey_col = self._find_column(merged_df, ['Stress_Level', 'Stress Level'])
            wearable_col = self._find_column(merged_df, ['Predicted Stress'])
            if not survey_col: raise ValueError("Could not find 'Stress_Level' column in survey CSV.")
            if not wearable_col: raise ValueError("Could not find 'Predicted Stress' column in wearable CSV.")

            stress_mapping = {'No stress': 'Low', 'Low Stress': 'Low', 'Medium stress': 'Medium', 'Medium Stress': 'Medium', 'High stress': 'High', 'High Stress': 'High'}
            merged_df['Survey Stress'] = merged_df[survey_col].map(stress_mapping)
            merged_df['Wearable Stress'] = merged_df[wearable_col].map(stress_mapping)

            # --- Generate Figures ---
            self.agreement_score = self._calculate_agreement_score(merged_df)
            self.cm_fig = self._plot_confusion_matrix(merged_df)
            self.dist_fig = self._plot_distributions(merged_df)
            self.box_fig = self._plot_biometric_boxplots(merged_df, ['EDA', 'TEMP'])
            self.corr_fig = self._plot_correlation_heatmap(merged_df)
            
            self.root.after(0, self._display_report)
            self.root.after(0, self._reset_ui, True)
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred: {e}")
            self.root.after(0, self._reset_ui, False)
    
    def _update_status(self, message, delay=0.5):
        self.root.after(0, self.status_label.config, {"text": message})
        time.sleep(delay)

    def _find_column(self, df, options):
        for col in options:
            if col.lower() in (c.lower() for c in df.columns):
                return [c for c in df.columns if c.lower() == col.lower()][0]
        return None
        
    def _calculate_agreement_score(self, data):
        valid_comp = data[['Survey Stress', 'Wearable Stress']].dropna()
        if not valid_comp.empty:
            return (valid_comp['Survey Stress'] == valid_comp['Wearable Stress']).mean() * 100
        return 0
    
    def _plot_confusion_matrix(self, data):
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=CARD_COLOR)
        order = ['Low', 'Medium', 'High']
        valid_comp = data[['Survey Stress', 'Wearable Stress']].dropna()
        cm = confusion_matrix(valid_comp['Survey Stress'], valid_comp['Wearable Stress'], labels=order)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=order, yticklabels=order, ax=ax, annot_kws={"size": 12})
        ax.set_title('Survey vs. Wearable Classification', fontsize=14, weight='bold', color=LABEL_COLOR)
        ax.set_xlabel('Predicted (Wearable)', fontsize=12, color=LABEL_COLOR)
        ax.set_ylabel('Actual (Survey)', fontsize=12, color=LABEL_COLOR)
        ax.tick_params(colors=LABEL_COLOR)
        plt.tight_layout()
        return fig
        
    def _plot_distributions(self, data):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor=CARD_COLOR)
        order = ['Low', 'Medium', 'High']
        colors = {'Low': '#22c55e', 'Medium': '#f59e0b', 'High': '#ef4444'}
        
        survey_counts = data['Survey Stress'].value_counts().reindex(order, fill_value=0)
        axes[0].pie(survey_counts, labels=survey_counts.index, autopct='%1.1f%%', startangle=90, colors=[colors.get(key) for key in survey_counts.index], wedgeprops=dict(width=0.4, edgecolor=CARD_COLOR), pctdistance=0.8, textprops={'color': LABEL_COLOR})
        axes[0].set_title('Survey Distribution', fontsize=14, weight='bold', color=LABEL_COLOR)

        wearable_counts = data['Wearable Stress'].value_counts().reindex(order, fill_value=0)
        axes[1].pie(wearable_counts, labels=wearable_counts.index, autopct='%1.1f%%', startangle=90, colors=[colors.get(key) for key in wearable_counts.index], wedgeprops=dict(width=0.4, edgecolor=CARD_COLOR), pctdistance=0.8, textprops={'color': LABEL_COLOR})
        axes[1].set_title('Wearable Distribution', fontsize=14, weight='bold', color=LABEL_COLOR)
        plt.tight_layout()
        return fig

    def _plot_biometric_boxplots(self, data, columns):
        valid_cols = [c for c in columns if c in data.columns]
        if not valid_cols: return None

        fig, axes = plt.subplots(1, len(valid_cols), figsize=(8, 4), facecolor=CARD_COLOR, squeeze=False)
        order = ['Low', 'Medium', 'High']
        for i, col in enumerate(valid_cols):
            ax = axes[0, i]
            sns.boxplot(x='Survey Stress', y=col, data=data, order=order, ax=ax)
            ax.set_title(f'{col} vs. Survey Stress', fontsize=14, weight='bold', color=LABEL_COLOR)
            ax.set_xlabel('Survey Stress Level', fontsize=12, color=LABEL_COLOR)
            ax.set_ylabel(col, fontsize=12, color=LABEL_COLOR)
            ax.tick_params(colors=LABEL_COLOR)
        plt.tight_layout()
        return fig

    def _plot_correlation_heatmap(self, data):
        total_col = self._find_column(data, ['Total', 'total_score'])
        if not total_col: return None
        
        corr_cols = [total_col, 'EDA', 'TEMP', 'EMG', 'RESP', 'ECG']
        valid_cols = [c for c in corr_cols if c in data.columns]
        if len(valid_cols) < 2: return None

        fig, ax = plt.subplots(figsize=(7, 5), facecolor=CARD_COLOR)
        sns.heatmap(data[valid_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Biometric & Survey Score Correlations', fontsize=14, weight='bold', color=LABEL_COLOR)
        ax.tick_params(colors=LABEL_COLOR)
        plt.tight_layout()
        return fig

    def _display_report(self):
        self._update_status("Report generated successfully.", delay=0)
        
        # --- Create a 2-column grid for some cards ---
        grid_frame = ttk.Frame(self.results_frame, style="TFrame")
        grid_frame.pack(fill="x", expand=True)
        grid_frame.columnconfigure((0, 1), weight=1)

        # --- Summary Card ---
        summary_card = ttk.Frame(grid_frame, style="Card.TFrame", padding=20)
        summary_card.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)
        ttk.Label(summary_card, text="Overall Agreement", style="Card.TLabel", font=(FONT_FAMILY, 16, "bold"), anchor="center").pack(fill="x")
        ttk.Label(summary_card, text=f"{self.agreement_score:.1f}%", style="Card.TLabel", font=(FONT_FAMILY, 48, "bold"), foreground=PRIMARY_COLOR, anchor="center").pack(fill="x")

        # --- Confusion Matrix in the grid ---
        self._add_plot_to_card(grid_frame, self.cm_fig).grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # --- Display remaining plots in a single column ---
        self._add_plot_to_card(self.results_frame, self.dist_fig).pack(fill="x", padx=10, pady=10, expand=True)
        if self.box_fig:
            self._add_plot_to_card(self.results_frame, self.box_fig).pack(fill="x", padx=10, pady=10, expand=True)
        if self.corr_fig:
            self._add_plot_to_card(self.results_frame, self.corr_fig).pack(fill="x", padx=10, pady=10, expand=True)
        
    def _add_plot_to_card(self, parent, figure):
        card = ttk.Frame(parent, style="Card.TFrame", padding=15)
        if figure is None:
            return card

        canvas = FigureCanvasTkAgg(figure, master=card)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)
        self.plot_canvases.append(canvas)
        return card

    def _reset_ui(self, success=False):
        self.analyze_button.config(state=tk.NORMAL, text="Analyze Data")
        if not success:
            self.status_label.config(text="An error occurred. Please check files and try again.")
        else:
            self.status_label.config(text="Report generated successfully.")

if __name__ == '__main__':
    root = tk.Tk()
    app = StressAnalysisApp(root)
    root.mainloop()

