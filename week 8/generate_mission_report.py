import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os

# --- Configuration ---
TEAM_DETAILS = {
    "Group Name": "The Closer",
    "Member 1 Name": "Rongala Sreedhar",
    "Member 1 Email": "rongalasreedhar@gmail.com",
    "Member 1 Country": "Italy",
    "Member 1 College/Company": "University of Naples Federico II",
    "Member 1 Specialization": "Data Science"
}
GITHUB_LINK = "https://github.com/example/repo" # Placeholder as not provided

FILE_PATH = "Healthcare_dataset (1).xlsx"
OUTPUT_PDF = "submission.pdf"

# --- Analysis Functions ---

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    # Load 'Dataset' sheet if it exists, otherwise default (or check logic)
    xl = pd.ExcelFile(filepath)
    if 'Dataset' in xl.sheet_names:
        return pd.read_excel(filepath, sheet_name='Dataset')
    else:
        # Fallback to finding sheet with most rows/cols or just the first one
        # For this specific task, we prioritize 'Dataset'
        return pd.read_excel(filepath)

def analyze_data(df):
    analysis = {}
    
    # 1. Data Understanding
    analysis['shape'] = df.shape
    analysis['columns'] = df.columns.tolist()
    analysis['dtypes'] = df.dtypes.astype(str).to_dict()
    
    # 2. Problems
    # Missing Values
    missing = df.isnull().sum()
    analysis['missing_values'] = missing[missing > 0].to_dict()
    
    # Duplicates
    analysis['duplicates'] = df.duplicated().sum()
    
    # Outliers (using IQR for numeric columns)
    outliers_report = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers_count > 0:
            outliers_report[col] = outliers_count
            
    analysis['outliers'] = outliers_report
    
    # Skewness
    skewness = df[numeric_cols].skew().to_dict()
    # Filter highly skewed
    analysis['skewness'] = {k: v for k, v in skewness.items() if abs(v) > 1}
    
    return analysis

def create_visualizations(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plots = []
    
    if len(numeric_cols) > 0:
        # Create a directory for plots if not exists relative to script
        # But here we just save to current dir and clean up or keep
        
        # Heatmap of missing values
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.savefig('missing_heatmap.png')
        plots.append('missing_heatmap.png')
        plt.close()

        # Distribution of first few numeric columns (limit to avoid too many pages)
        for col in numeric_cols[:3]: 
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            filename = f'dist_{col}.png'.replace(" ", "_").replace("/", "_")
            plt.savefig(filename)
            plots.append(filename)
            plt.close()
            
    return plots

# --- PDF Generation ---

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Data Analysis Project Submission', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()

def generate_pdf(analysis, plots):
    pdf = PDFReport()
    pdf.add_page()
    
    # Team Details
    pdf.chapter_title("Team Member's Details")
    details_text = ""
    for k, v in TEAM_DETAILS.items():
        details_text += f"{k}: {v}\n"
    pdf.chapter_body(details_text)
    
    # Problem Description
    pdf.chapter_title("Problem Description")
    pdf.chapter_body(
        "The goal is to analyze the 'Healthcare_dataset' to identify data quality issues "
        "such as missing values, outliers, and skewness, and to propose appropriate strategies "
        "for data cleaning and preprocessing to prepare the dataset for further modeling tasks (e.g., specific classification or regression)."
    )
    
    # Data Understanding
    pdf.chapter_title("Data Understanding")
    data_text = (
        f"Dataset: {FILE_PATH}\n"
        f"Shape: {analysis['shape'][0]} rows, {analysis['shape'][1]} columns\n\n"
        "Type of Data: Structured / Tabular (Healthcare records).\n"
        "Key Features types:\n"
    )
    # Simplify dtypes for report
    for col, dtype in list(analysis['dtypes'].items())[:10]: # Limit list
        data_text += f"- {col}: {dtype}\n"
    if len(analysis['dtypes']) > 10:
        data_text += "...(and more)\n"
        
    pdf.chapter_body(data_text)
    
    # Problems in Data
    pdf.chapter_title("Problems in the Data")
    
    # Missing
    if analysis['missing_values']:
        missing_text = "Missing Values found:\n"
        for k, v in analysis['missing_values'].items():
            missing_text += f"- {k}: {v} missing\n"
    else:
        missing_text = "No missing values found.\n"
    
    # Outliers
    if analysis['outliers']:
        outliers_text = "\nOutliers detected (using IQR method):\n"
        for k, v in analysis['outliers'].items():
            outliers_text += f"- {k}: {v} potential outliers\n"
    else:
        outliers_text = "\nNo significant outliers detected by IQR.\n"
        
    # Skewness
    if analysis['skewness']:
        skew_text = "\nSkewed Features (|skew| > 1):\n"
        for k, v in analysis['skewness'].items():
            skew_text += f"- {k}: {v:.2f}\n"
    else:
        skew_text = "\nNo highly skewed features.\n"
        
    pdf.chapter_body(missing_text + outliers_text + skew_text)
    
    # Visualizations
    if plots:
        pdf.add_page()
        pdf.chapter_title("Visualizations")
        for plot_file in plots:
            if os.path.exists(plot_file):
                pdf.image(plot_file, w=170)
                pdf.ln(5)
                # os.remove(plot_file) # Clean up later manually or keep
    
    # Approaches
    pdf.add_page()
    pdf.chapter_title("Approaches to Handle Data Problems")
    approaches_text = (
        "1. Missing Values:\n"
        "   - Numerical: Impute with Median (robust to outliers) or Mean (if normal).\n"
        "   - Categorical: Impute with Mode or create a 'Missing' category.\n"
        "   - Rationale: Preserves data volume compared to dropping rows.\n\n"
        "2. Outliers:\n"
        "   - Log Transformation: To reduce the impact of extreme values.\n"
        "   - Capping (Winsorization): Limiting extreme values to the 1st/99th percentiles.\n"
        "   - Rationale: Linear models are sensitive to outliers; these methods reduce their influence without losing data points.\n\n"
        "3. Skewness:\n"
        "   - Apply Log or Box-Cox transformations to normalize distributions.\n"
        "   - Rationale: Many statistical models assume normality."
    )
    pdf.chapter_body(approaches_text)
    
    # Github Link
    pdf.chapter_title("GitHub Repo Link")
    pdf.chapter_body(GITHUB_LINK)
    
    pdf.output(OUTPUT_PDF)
    print(f"PDF generated: {OUTPUT_PDF}")

def main():
    print("Loading data...")
    try:
        df = load_data(FILE_PATH)
    except Exception as e:
        print(e)
        return

    print("Analyzing data...")
    analysis = analyze_data(df)
    
    print("Creating visualizations...")
    plots = create_visualizations(df)
    
    print("Generating PDF...")
    generate_pdf(analysis, plots)
    
    # Cleanup plots
    for p in plots:
        try:
            if os.path.exists(p):
                os.remove(p)
        except:
            pass

if __name__ == "__main__":
    main()
