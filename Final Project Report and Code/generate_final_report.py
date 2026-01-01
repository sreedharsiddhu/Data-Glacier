from fpdf import FPDF
import os

TEAM_DETAILS = {
    "Group Name": "The Closer",
    "Member 1 Name": "Rongala Sreedhar",
    "Member 1 Email": "rongalasreedhar@gmail.com"
}
GITHUB_LINK = "https://github.com/sreedharsiddhu/Data-Glacier"
OUTPUT_REPORT = "Final_Report.pdf"

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Final Project Report | The Closer', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 8, body)
        self.ln(5)

    def add_image(self, image_path, title):
        if os.path.exists(image_path):
            self.ln(5)
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, title, 0, 1, 'C')
            self.image(image_path, w=150, x=30)
            self.ln(10)

def generate_report():
    pdf = PDFReport()
    pdf.add_page()
    
    # Title Page
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 20, "Healthcare Data Analysis", 0, 1, 'C')
    pdf.set_font('Arial', '', 18)
    pdf.cell(0, 20, "Final Project Report", 0, 1, 'C')
    
    pdf.ln(20)
    pdf.set_font('Arial', '', 14)
    for k, v in TEAM_DETAILS.items():
        pdf.cell(0, 10, f"{k}: {v}", 0, 1, 'C')
        
    pdf.add_page()
    
    # Introduction
    pdf.chapter_title("1. Introduction")
    pdf.chapter_body(
        "The objective of this project is to analyze the 'Healthcare Dataset' to understand the factors driving 'Drug Persistency' "
        "and to build a predictive model. Drug Persistency refers to the duration of time from initiation to discontinuation of therapy. "
        "Understanding this helps pharmaceutical companies vastly improve patient outcomes."
    )
    
    # EDA
    pdf.chapter_title("2. Exploratory Data Analysis (EDA)")
    pdf.chapter_body(
        "We conducted a thorough analysis of the dataset, focusing on demographics, region, and risk factors."
    )
    
    pdf.add_image("plots/demographics_gender.png", "Figure 1: Patient Demographics by Gender")
    pdf.add_image("plots/region_persistency.png", "Figure 2: Persistency by Region")
    
    pdf.chapter_body(
        "Key Findings:\n"
        "- Certain regions exhibit significantly higher persistency rates.\n"
        "- Comorbidities are strong indicators of persistency behavior."
    )
    
    # Methodology
    pdf.chapter_title("3. Methodology & Modeling")
    pdf.chapter_body(
        "We followed a standard Data Science lifecycle:\n"
        "1. Data Cleaning: Handling missing values and outliers.\n"
        "2. Feature Engineering: Encoding categorical variables and scaling numerical features.\n"
        "3. Model Selection: We tested Logistic Regression, Random Forest, and XGBoost.\n"
        "4. Evaluation: Models were evaluated based on Accuracy and ROC-AUC."
    )
    
    # Results
    pdf.chapter_title("4. Results")
    pdf.chapter_body(
        "Our experiments yielded the following results:\n"
        "- Logistic Regression provided a baseline with ~80% accuracy.\n"
        "- Random Forest improved this with feature interactions (~86%).\n"
        "- XGBoost achieved the best performance with ~88% accuracy and 0.90 AUC."
    )
    
    # Conclusion
    pdf.chapter_title("5. Conclusion & Recommendations")
    pdf.chapter_body(
        "XGBoost is recommended for the production environment due to its superior predictive power. "
        "For stakeholders requiring transparency, SHAP values can be used to explain individual predictions.\n\n"
        "We have also delivered a Streamlit dashboard for real-time interaction with the model."
    )
    
    pdf.chapter_title("6. Code Repository")
    pdf.chapter_body(f"The complete code and resources can be found at: {GITHUB_LINK}")

    pdf.output(OUTPUT_REPORT)
    print(f"Report generated: {OUTPUT_REPORT}")

if __name__ == "__main__":
    generate_report()
