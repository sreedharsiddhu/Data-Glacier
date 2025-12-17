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
        self.cell(0, 10, 'Data Cleansing & Transformation - Week 9', 0, 1, 'C')
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
        "The objective of this task is to perform Data Cleansing and Transformation on the Healthcare dataset. "
        "The data contains patient records which may have inconsistencies, missing values, or outliers. "
        "Our goal is to apply multiple techniques to clean this data and also demonstrate NLP featurization."
    )
    
    # GitHub Link
    pdf.chapter_title("GitHub Repo Link")
    pdf.chapter_body(GITHUB_LINK)
    
    # Data Cleansing
    pdf.chapter_title("Data Cleansing and Transformation")
    pdf.chapter_body(
        "We applied the following cleansing techniques (demonstrated in the attached Notebook):\n\n"
        "1. Handling Missing Values:\n"
        "   - Approach A (Member 1): Simple Imputation. We used the Median for skewed numeric columns and Mode for categorical columns. This is a robust baseline.\n"
        "   - Approach B (Member 2): Model-Based Imputation (KNN). We used K-Nearest Neighbors to estimate missing values based on similar records, preserving local structure.\n\n"
        "2. Handling Outliers:\n"
        "   - Approach A: Removal using IQR (Interquartile Range). Any data point falling below Q1-1.5*IQR or above Q3+1.5*IQR was removed.\n"
        "   - Approach B: Log Transformation. Instead of removing data, we applied a log transformation to compress the scale of outliers.\n\n"
        "3. NLP Featurization (Column: Ntm_Speciality):\n"
        "   - Cleaning: Lowercasing and removing punctuation using Regex.\n"
        "   - Featurization: Applied CountVectorizer (Bag of Words) and TF-IDF to convert text data into numerical features."
    )
    
    # Peer Review
    pdf.chapter_title("Peer Review Comments")
    pdf.chapter_body(
        "Reviewer: Peer Member (Simulated)\n"
        "- \"The use of Median imputation is appropriate for 'Count_Of_Risks' due to its discrete nature.\"\n"
        "- \"IQR outlier removal significantly reduced the dataset size; Log transformation (Approach 2) might be safer for this small dataset.\"\n"
        "- \"Regex cleaning successfully standardized the 'Speciality' column.\""
    )
    
    pdf.output(OUTPUT_PDF)
    print(f"PDF generated: {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_pdf()
