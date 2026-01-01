import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ensure output directory for plots exists
if not os.path.exists('plots'):
    os.makedirs('plots')

df = pd.read_excel('Healthcare_dataset.xlsx', sheet_name='Dataset')

sns.set_palette("pastel")

# 1. Demographics: Gender Distribution (Pie Chart)
plt.figure(figsize=(6, 6))
df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Patient Demographics by Gender', fontsize=14)
plt.ylabel('')
plt.tight_layout()
plt.savefig('plots/demographics_gender.png')
plt.close()

# 2. Regional Analysis: Persistency by Region
plt.figure(figsize=(10, 6))
sns.countplot(x='Region', hue='Persistency_Flag', data=df)
plt.title('Drug Persistency by Region', fontsize=14)
plt.ylabel('Count of Patients')
plt.xlabel('Region')
plt.legend(title='Persistency')
plt.tight_layout()
plt.savefig('plots/region_persistency.png')
plt.close()

# 3. Top Risk Factors
# Identifying columns that start with 'Risk_'
risk_cols = [col for col in df.columns if col.startswith('Risk_')]
risk_counts = df[risk_cols].apply(lambda x: x.value_counts()).loc['Y'].sort_values(ascending=False).head(5)

plt.figure(figsize=(10, 6))
sns.barplot(x=risk_counts.values, y=risk_counts.index, palette='viridis')
plt.title('Top 5 Medical Risk Factors', fontsize=14)
plt.xlabel('Number of Patients with Risk')
plt.tight_layout()
plt.savefig('plots/top_risks.png')
plt.close()

# 4. Age Bucket Distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='Age_Bucket', hue='Persistency_Flag', data=df, order=sorted(df['Age_Bucket'].unique()))
plt.title('Persistency across Age Groups', fontsize=14)
plt.tight_layout()
plt.savefig('plots/age_distribution.png')
plt.close()

print("Plots generated in 'plots/' directory.")
