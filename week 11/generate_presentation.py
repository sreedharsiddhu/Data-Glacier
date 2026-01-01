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

OUTPUT_PDF = "presentation.pdf"

class PDFPresentation(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'The Closer | Week 11 EDA Presentation | Slide {self.page_no()}', 0, 0, 'C')

def create_slide(pdf, title, content=None, image_path=None):
    pdf.add_page(orientation='L') # Landscape
    
    # Title
    pdf.set_font('Arial', 'B', 24)
    pdf.set_fill_color(44, 62, 80) # Dark Blue
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 20, title, 0, 1, 'C', 1)
    
    # Reset color
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Content (Text)
    if content:
        pdf.set_font('Arial', '', 14)
        pdf.multi_cell(0, 8, content)
        pdf.ln(5)
        
    # Image
    if image_path and os.path.exists(image_path):
        # Center image
        img_width = 160
        x = (297 - img_width) / 2 # A4 Landscape width is 297mm
        pdf.image(image_path, x=x, w=img_width)

def generate_presentation():
    pdf = PDFPresentation()
    
    # Slide 1: Title Slide
    pdf.add_page(orientation='L')
    pdf.set_fill_color(236, 240, 241)
    pdf.rect(0, 0, 297, 210, 'F') # Background
    
    pdf.set_y(60)
    pdf.set_font('Arial', 'B', 36)
    pdf.cell(0, 20, "Healthcare Data Analysis", 0, 1, 'C')
    pdf.set_font('Arial', '', 24)
    pdf.cell(0, 20, "EDA & Strategic Recommendations", 0, 1, 'C')
    
    pdf.set_y(120)
    pdf.set_font('Arial', '', 14)
    for k, v in TEAM_DETAILS.items():
        pdf.cell(0, 8, f"{k}: {v}", 0, 1, 'C')
        
    # Slide 2: Problem Description
    create_slide(pdf, "Problem Description", 
                 "We aim to identify the key factors driving 'Drug Persistency' in patients. "
                 "Understanding why patients discontinue therapy is crucial for improving health outcomes and pharmaceutical strategy. "
                 "\n\nOur goal is to analyze the 'Healthcare Dataset' to detect patterns in demographics, regionality, and comorbidity risks.")

    # Slide 3: Demographics
    create_slide(pdf, "Patient Demographics", 
                 "Understanding our patient base by Gender.", 
                 "plots/demographics_gender.png")

    # Slide 4: Regional Analysis
    create_slide(pdf, "Regional Trends", 
                 "Distribution of Persistent vs Non-Persistent patients across different regions. This helps pinpoint geographical areas for targeted intervention.", 
                 "plots/region_persistency.png")

    # Slide 5: Risk Factors
    create_slide(pdf, "Top Comorbidities & Risks", 
                 "Analysis of the most prevalent risk factors recorded in the dataset. These are potential key drivers for the model.", 
                 "plots/top_risks.png")
                 
    # Slide 6: Age Analysis
    create_slide(pdf, "Age Group Analysis", 
                 "Persistency rates across different age buckets. Are older patients more likely to adhere to therapy?", 
                 "plots/age_distribution.png")

    # Slide 7: GitHub Link
    create_slide(pdf, "Project Repository", 
                 f"All code, analysis, and reports are available at:\n\n{GITHUB_LINK}")

    # Slide 8: Technical Recommendation (Last Slide)
    create_slide(pdf, "Technical Recommendations (Model Selection)")
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, 
        "Recommended Modeling Strategy for Data Science Team:\n\n"
        "1. Baseline Model: Logistic Regression\n"
        "   - Pros: Highly interpretable, provides odds ratios for risk factors.\n"
        "   - Use case: Communicating 'Why' to stakeholders.\n\n"
        "2. Production Model: XGBoost / LightGBM\n"
        "   - Pros: Handles non-linear relationships, missing values, and feature interactions better than linear models.\n"
        "   - Performance: Expected to yield higher AUC-ROC scores.\n\n"
        "3. Validation Strategy:\n"
        "   - Stratified K-Fold Cross Validation (to handle any class imbalance).\n"
        "   - Metrics: F1-Score (balance precision/recall) and ROC-AUC.\n\n"
        "4. Feature Engineering Plan:\n"
        "   - Create 'Comorbidity_Index' (sum of all Risk flags).\n"
        "   - Target Encode high-cardinality categorical variables (e.g., 'Speciality')."
    )

    pdf.output(OUTPUT_PDF)
    print(f"Presentation generated: {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_presentation()
