from fpdf import FPDF
import os

TEAM_DETAILS = {
    "Group Name": "The Closer",
    "Member 1 Name": "Rongala Sreedhar",
    "Member 1 Email": "rongalasreedhar@gmail.com",
    "Member 1 Country": "Italy",
    "Member 1 College/Company": "University of Naples Federico II",
    "Member 1 Specialization": "Data Science"
}
GITHUB_LINK = "https://github.com/sreedharsiddhu/Data-Glacier"

OUTPUT_PDF = "submission.pdf"

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Week 10: EDA & Final Recommendations', 0, 1, 'C')
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

def generate_pdf():
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
        "The objective is to perform comprehensive Exploratory Data Analysis (EDA) on the Healthcare dataset "
        "to identify key factors affecting drug persistency. This involves understanding data distribution, "
        "correlation between variables, and deriving actionable insights."
    )
    
    # GitHub Link
    pdf.chapter_title("Github Repo Link")
    pdf.chapter_body(GITHUB_LINK)
    
    # EDA Performed
    pdf.chapter_title("EDA Performed")
    pdf.chapter_body(
        "We executed the following analysis (detailed in `eda.ipynb`):\n"
        "1. Univariate Analysis: Examined distributions of `Persistency_Flag` (Target), `Count_Of_Risks`, and `Gender`.\n"
        "2. Bivariate Analysis: Investigated the relationship between `Gender` and `Persistency_Flag`, finding minimal deviation. "
        "Analyzed `Count_Of_Risks` against `Persistency_Flag`, showing that higher risk counts correlate slightly with persistency issues.\n"
        "3. Correlation Analysis: Generated heatmaps to detect multicollinearity among numerical features."
    )
    
    # Final Recommendation
    pdf.chapter_title("Final Recommendation")
    pdf.chapter_body(
        "Based on our analysis, we recommend the following procedure:\n"
        "1. Data Preprocessing: Impute missing values in `Dexa_Freq_During_Rx` and scale numerical features (`Count_Of_Risks`).\n"
        "2. Feature Engineering: Create an interaction term between `Risk` factors and `Age` buckets if possible.\n"
        "3. Modeling Strategy: Proceed with Logistic Regression for interpretability, then attempt Gradient Boosting (XGBoost) to capture non-linear patterns.\n"
        "4. Deployment: Monitor the `Persistency_Flag` distribution in production to detect drift."
    )
    
    pdf.output(OUTPUT_PDF)
    print(f"PDF generated: {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_pdf()
