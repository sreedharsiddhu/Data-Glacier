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

OUTPUT_PDF = "Final_Presentation.pdf"

class PDFPresentation(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'The Closer | Final Project Presentation | Slide {self.page_no()}', 0, 0, 'C')

def create_slide(pdf, title, content=None, bullets=None, image_path=None):
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
    y_start = pdf.get_y()
    
    if content:
        pdf.set_font('Arial', '', 14)
        pdf.multi_cell(0, 8, content)
        pdf.ln(5)
    
    if bullets:
        pdf.set_font('Arial', '', 14)
        for bullet in bullets:
            pdf.cell(10) # Indent
            pdf.cell(0, 8, f"- {bullet}", 0, 1)
        pdf.ln(5)

    # Image
    if image_path and os.path.exists(image_path):
        # Place image on the right or bottom depending on space
        img_width = 130
        x = (297 - img_width) / 2 # Center
        if content or bullets:
            y = pdf.get_y()
            if y < 120:
                y = 120
            pdf.image(image_path, x=x, y=y, w=img_width)
        else:
             pdf.image(image_path, x=x, y=60, w=img_width)

def generate_presentation():
    pdf = PDFPresentation()
    
    # Slide 1: Title Slide
    pdf.add_page(orientation='L')
    pdf.set_fill_color(236, 240, 241)
    pdf.rect(0, 0, 297, 210, 'F')
    
    pdf.set_y(60)
    pdf.set_font('Arial', 'B', 36)
    pdf.cell(0, 20, "Healthcare Data Analysis", 0, 1, 'C')
    pdf.set_font('Arial', '', 24)
    pdf.cell(0, 20, "Predicting Drug Persistency", 0, 1, 'C')
    
    pdf.set_y(120)
    pdf.set_font('Arial', '', 14)
    for k, v in TEAM_DETAILS.items():
        pdf.cell(0, 8, f"{k}: {v}", 0, 1, 'C')
        
    # Slide 2: Project Overview
    create_slide(pdf, "Project Overview", 
                 "Objective: To analyze factors affecting Drug Persistency and build a predictive model.",
                 bullets=[
                     "Understanding patient demographics and comorbidities.",
                     "Identifying key risk factors.",
                     "Developing a Machine Learning model to predict persistency.",
                     "Creating an interactive dashboard for stakeholders."
                 ])

    # Slide 3: EDA - Demographics
    create_slide(pdf, "EDA: Patient Demographics", 
                 "Key Insight: Gender distribution shows slight variation.", 
                 image_path="plots/demographics_gender.png")

    # Slide 4: EDA - Regional Analysis
    create_slide(pdf, "EDA: Regional Trends", 
                 "Key Insight: Significant regional disparities in persistency rates.", 
                 image_path="plots/region_persistency.png")

    # Slide 5: EDA - Risk Factors
    create_slide(pdf, "EDA: Top Risk Factors", 
                 "Key Insight: Comorbidities heavily influence persistency.", 
                 image_path="plots/top_risks.png")

    # Slide 6: Model Strategy
    create_slide(pdf, "Modeling Strategy", 
                 "We explored three families of models to balance accuracy and interpretability:",
                 bullets=[
                     "Linear Model: Logistic Regression (Baseline, High Interpretability).",
                     "Ensemble Model: Random Forest (Feature Importance).",
                     "Boosting Model: XGBoost (High Accuracy)."
                 ])

    # Slide 7: Model Results (Detailed)
    create_slide(pdf, "Model Performance Comparison",
                 "We evaluated models based on ROC-AUC and Confusion Matrices.",
                 image_path="plots/model_roc_curve.png")

    # Slide 8: Confusion Matrix (Random Forest)
    create_slide(pdf, "Confusion Matrix: Random Forest",
                 "Random Forest showed a strong balance between Precision and Recall.",
                 image_path="plots/cm_random_forest.png")

    # Slide 9: Feature Importance
    create_slide(pdf, "Feature Importance",
                 "Top drivers of persistency as identified by the Random Forest model.",
                 image_path="plots/feature_importance.png")

    # Slide 10: Explainability & Dashboard
    create_slide(pdf, "Interactive Dashboard", 
                 "A Streamlit dashboard was developed to allow stakeholders to:",
                 bullets=[
                     "Visualize dataset statistics.",
                     "Input patient details.",
                     "Get real-time persistency predictions."
                 ])

    # Slide 11: Conclusion
    create_slide(pdf, "Conclusion & Next Steps", 
                 bullets=[
                     "Data quality improved through preprocessing.",
                     "Robust predictive model achieved.",
                     "Actionable insights on regional and risk-based targeting.",
                     "Next Steps: Deploy dashboard and monitor model drift."
                 ])
    
    # Slide 12: Links
    create_slide(pdf, "Code & Deliverables", 
                  f"Full Source Code and Report available at:\n\n{GITHUB_LINK}")

    pdf.output(OUTPUT_PDF)
    print(f"Presentation generated: {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_presentation()
