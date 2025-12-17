# Project Submission Template

## Team Member's Details
- Group Name: 
- Member 1 Name: 
- Member 1 Email: 
- Member 1 Country: 
- Member 1 College/Company: 
- Member 1 Specialization (Data Science / NLP / Data Analyst): 

- Member 2 Name: 
- Member 2 Email: 
- Member 2 Country: 
- Member 2 College/Company: 
- Member 2 Specialization: 

(Repeat for additional members)

---

## Problem Description
(Describe the problem you are addressing — business context, goals, and desired outcomes.)


## Data Understanding
(Describe the dataset: source, how it was collected, number of rows, number of features, feature descriptions.)


## Type of Data for Analysis
- Structured / Tabular: 
- Text: 
- Time series: 
- Images: 
- Mixed: 
- Other: 

Provide details about key variables and sample rows if helpful.


## Problems in the Data
- Missing / NA values: (list columns and counts)
- Outliers: (which features show outliers and how many)
- Skewness: (which features are skewed and direction)
- Class imbalance (for classification tasks): (class counts)
- Duplicates: (rows/records duplicated)
- Inconsistent formats / encoding issues: 
- Other issues: 

Include a small table or bullet list with counts where available.


## Approaches to Handle Data Problems (and Rationale)
- Missing values:
  - Option A: Remove rows with missing values — when missingness is rare and random.
  - Option B: Impute with mean/median/mode — for numeric features with low missing rate.
  - Option C: Model-based imputation (KNN, iterative) — when preserving variance matters.
  - Option D: Flag missingness with an indicator column — when missingness is informative.

- Outliers:
  - Option A: Winsorization / clipping — to reduce effect of extreme values on models.
  - Option B: Log or power transforms — to reduce skew and compress large values.
  - Option C: Remove extreme outliers after domain validation — when values are erroneous.

- Skewed distributions:
  - Log / Box-Cox / Yeo-Johnson transforms — to normalize distributions for linear models.

- Categorical encoding:
  - One-hot encoding for low-cardinality features.
  - Target / mean encoding for high-cardinality features.

- Class imbalance:
  - Resampling (SMOTE, ADASYN) or class-weighted models.

- Feature scaling:
  - StandardScaler / MinMaxScaler for models sensitive to scale.

Explain why you choose each approach (trade-offs and suitability for chosen models).


## GitHub Repo Link
(Provide the link to your repository, for example: https://github.com/your-org/your-repo)


---

## Optional: Brief Project Plan and Timeline
- Data cleaning: 
- Exploratory analysis: 
- Feature engineering: 
- Modeling & validation: 
- Presentation & deliverables: 


---

Please fill in the team details and dataset-specific numbers. Once you confirm the filled content, I can convert this into a PDF and save `submission.pdf` in the workspace for you.