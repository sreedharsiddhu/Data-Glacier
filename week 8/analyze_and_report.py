import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import json
import subprocess


def find_first_excel(dirpath: Path):
    exts = ["*.xlsx", "*.xls"]
    for ext in exts:
        files = list(dirpath.glob(ext))
        if files:
            return files[0]
    return None


def create_output_dir(base: Path):
    out = base / "report_output"
    out.mkdir(exist_ok=True)
    return out


def summarize_dataframe(df: pd.DataFrame):
    summary = {}
    summary['rows'], summary['columns'] = df.shape
    summary['dtypes'] = df.dtypes.astype(str).to_dict()
    summary['head'] = df.head(5).to_dict(orient='records')
    summary['missing'] = df.isna().sum().to_dict()
    summary['duplicates'] = int(df.duplicated().sum())
    return summary


def numeric_column_stats(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    stats = {}
    for col in num.columns:
        series = num[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        stats[col] = {
            'count': int(series.count()),
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'skewness': float(skew(series)) if series.size > 2 else None,
            'q1': float(q1),
            'q3': float(q3),
            'iqr': float(iqr),
            'outlier_count': int(outliers.count()),
            'outlier_fraction': float(outliers.count()) / float(series.count()) if series.count() else 0.0
        }
    return stats


def categorical_summary(df: pd.DataFrame, top_n=10):
    cat = df.select_dtypes(include=['object', 'category', 'bool'])
    cats = {}
    for col in cat.columns:
        vc = col in df and df[col].value_counts(dropna=False)
        if vc is not None:
            cats[col] = vc.head(top_n).to_dict()
    return cats


def plot_overview(df: pd.DataFrame, outdir: Path, max_plots=6):
    sns.set(style='whitegrid')
    num = df.select_dtypes(include=[np.number])
    plots = []
    # Histograms
    for i, col in enumerate(num.columns[:max_plots]):
        plt.figure(figsize=(6,4))
        sns.histplot(num[col].dropna(), kde=True)
        plt.title(f'Histogram: {col}')
        p = outdir / f'hist_{col}.png'
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        plots.append(p.name)

    # Boxplots for the same columns
    for i, col in enumerate(num.columns[:max_plots]):
        plt.figure(figsize=(6,3))
        sns.boxplot(x=num[col].dropna())
        plt.title(f'Boxplot: {col}')
        p = outdir / f'box_{col}.png'
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        plots.append(p.name)

    # Missingness heatmap
    try:
        plt.figure(figsize=(10,6))
        sns.heatmap(df.isna(), cbar=False)
        plt.title('Missingness Heatmap')
        p = outdir / 'missingness_heatmap.png'
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        plots.append(p.name)
    except Exception:
        pass

    return plots


def generate_markdown_report(summary, num_stats, cat_summary, plots, outdir: Path, dataset_name: str):
    md = []
    md.append(f"# Project Submission Report\n")
    md.append("## Team Member's Details\n- Group Name: \n- Member Names / Emails / Country / College / Specialization:\n")
    md.append("## Problem Description\n(Describe problem here)\n")
    md.append("## Data Understanding\n")
    md.append(f"- Dataset file: {dataset_name}\n")
    md.append(f"- Rows: {summary['rows']}, Columns: {summary['columns']}\n")
    md.append("### Column types\n")
    for k, v in summary['dtypes'].items():
        md.append(f"- {k}: {v}\n")
    md.append("\n### Missing values\n")
    for k, v in summary['missing'].items():
        if v:
            md.append(f"- {k}: {v}\n")
    md.append("\n### Duplicates\n")
    md.append(f"- Duplicate rows: {summary['duplicates']}\n")

    md.append("\n## Type of Data for Analysis\n")
    md.append("- Structured / Tabular: Yes\n")
    md.append("- Text / Other: See categorical summaries\n")

    md.append("\n## Problems in the Data\n")
    # numeric problems
    for col, s in num_stats.items():
        md.append(f"- {col}: missing={int(s['count']) < summary['rows']} | outliers={s['outlier_count']} ({s['outlier_fraction']:.2%}) | skewness={s['skewness']:.3f}\n")

    md.append("\n## Categorical summaries (top values)\n")
    for col, vc in cat_summary.items():
        md.append(f"- {col}: {vc}\n")

    md.append("\n## Plots\n")
    for p in plots:
        md.append(f"![{p}](report_output/{p})\n")

    md.append("\n## Approaches to Handle Data Problems\n")
    md.append("- Missing values: impute (mean/median) or remove rows depending on missing fraction.\n")
    md.append("- Outliers: investigate, then winsorize or remove if erroneous; consider transforms.\n")
    md.append("- Skewed distributions: log / box-cox / yeo-johnson transforms.\n")
    md.append("- Categorical encoding: one-hot for low cardinality; target encoding for high-cardinality.\n")
    md.append("\n## GitHub Repo Link\n(Provide your repo link here)\n")

    md.append("\n## Optional: Brief Project Plan and Timeline\n- Data cleaning: \n- Exploratory analysis: \n- Feature engineering: \n- Modeling & validation: \n- Presentation & deliverables: \n")

    md_file = outdir / 'report.md'
    md_text = '\n'.join(md)
    md_file.write_text(md_text, encoding='utf-8')
    return md_file


def try_pandoc_convert(md_file: Path, pdf_file: Path):
    try:
        subprocess.run(['pandoc', str(md_file), '-o', str(pdf_file)], check=True)
        return True
    except Exception:
        return False


def main():
    base = Path(__file__).parent
    data_file = find_first_excel(base)
    if data_file is None:
        print('No Excel dataset found in the script directory. Place your dataset in the same folder as this script.')
        sys.exit(1)

    print(f'Using dataset: {data_file.name}')
    outdir = create_output_dir(base)

    df = pd.read_excel(data_file)

    summary = summarize_dataframe(df)
    num_stats = numeric_column_stats(df)
    cat_summary = categorical_summary(df)
    plots = plot_overview(df, outdir)

    md_file = generate_markdown_report(summary, num_stats, cat_summary, plots, outdir, data_file.name)
    pdf_file = outdir / 'submission.pdf'

    converted = try_pandoc_convert(md_file, pdf_file)
    if converted:
        print(f'PDF generated at: {pdf_file}')
    else:
        print('Pandoc not available or conversion failed. Markdown report saved at:', md_file)
        print('You can convert the markdown to PDF with Pandoc:')
        print(f"pandoc \"{md_file}\" -o \"{pdf_file}\"")


if __name__ == '__main__':
    main()
