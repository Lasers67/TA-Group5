import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.utils import resample

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

# Define gender palette
gender_palette = {'F': '#FF69B4', 'M': '#1E90FF'}

# Define paths
script_dir = os.path.dirname(__file__) if __name__ == "__main__" else "."
results_dir = os.path.join(script_dir, "Results")
csv_path = os.path.join(results_dir, "metrics_results_all_languages.csv")

# Read the CSV file
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: '{csv_path}' not found.")
    exit(1)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

# Ensure required columns exist
required_columns = ['Language', 'Gender', 'MSL', 'FWF', 'FK']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns in CSV: {missing_columns}")
    exit(1)

# Clean data: Filter for Male and Female, handle missing/invalid values
df = df[df['Gender'].isin(['M', 'F'])].copy()
df = df[df['Language'].isin(['EN', 'ES', 'FR'])].copy()
df = df.dropna(subset=required_columns)
df[['MSL', 'FWF', 'FK']] = df[['MSL', 'FWF', 'FK']].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['MSL', 'FWF', 'FK'])

# Clip FK to reduce impact of extreme outliers (e.g., -1152)
df['FK_clipped'] = df['FK'].clip(lower=-200, upper=200)

# Melt data for Lexical and Linguistic Complexity (MSL, FK)
df_lexical = df.melt(id_vars=['Language', 'Gender'], value_vars=['MSL', 'FK_clipped'],
                     var_name='Metric', value_name='Value')
df_lexical['Metric'] = df_lexical['Metric'].replace({'FK_clipped': 'FK'})

# Compute means and confidence intervals for FWF using bootstrapping
def compute_bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    if len(data) < 2:
        return 0, 0
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = resample(data, replace=True)
        bootstrapped_means.append(np.mean(sample))
    lower = np.percentile(bootstrapped_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrapped_means, (1 + confidence) / 2 * 100)
    return lower, upper

fwf_stats = df.groupby(['Language', 'Gender'])['FWF'].agg([
    'mean',
    lambda x: compute_bootstrap_ci(x)[0],
    lambda x: compute_bootstrap_ci(x)[1]
]).reset_index()
fwf_stats.columns = ['Language', 'Gender', 'Mean', 'CI_Lower', 'CI_Upper']

# Ensure error bars are non-negative
fwf_stats['Err_Lower'] = fwf_stats['Mean'] - fwf_stats['CI_Lower']
fwf_stats['Err_Upper'] = fwf_stats['CI_Upper'] - fwf_stats['Mean']
fwf_stats['Err_Lower'] = fwf_stats['Err_Lower'].clip(lower=0)
fwf_stats['Err_Upper'] = fwf_stats['Err_Upper'].clip(lower=0)

# Visualization 1: Lexical and Linguistic Complexity (MSL, FK) - Violin Plot
plt.figure(figsize=(12, 8))
g = sns.FacetGrid(df_lexical, col="Metric", row="Language", height=3, aspect=1.5, sharey=False)
g.map(sns.violinplot, "Gender", "Value", split=True, palette=gender_palette, order=['M', 'F'])
g.set_titles("{row_name} - {col_name}")
g.set_axis_labels("Gender", "Value")
for ax in g.axes.flat:
    metric = ax.get_title().split(' - ')[1]
    if metric == 'MSL':
        ax.set_ylabel('Mean Sentence Length (Words)')
    elif metric == 'FK':
        ax.set_ylabel('Flesch-Kincaid Score (Clipped)')
    ax.set_xlabel('Gender')
g.add_legend(title="Gender")
plt.tight_layout()
plot1_path = os.path.join(results_dir, "lexical_linguistic_gender_metrics.png")
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Lexical and Linguistic Gender Metrics plot saved to '{plot1_path}'")

# Visualization 2: Gender-Based Writing Differences (FWF) - Grouped Bar Plot with Error Bars
plt.figure(figsize=(8, 6))
sns.barplot(x='Language', y='Mean', hue='Gender', data=fwf_stats, palette=gender_palette)
for i, row in fwf_stats.iterrows():
    x_pos = {'EN': 0, 'ES': 1, 'FR': 2}[row['Language']] + (-0.2 if row['Gender'] == 'M' else 0.2)
    plt.errorbar(x=x_pos, y=row['Mean'], yerr=[[row['Err_Lower']], [row['Err_Upper']]], fmt='none', c='black', capsize=5)
plt.title('Function Word Frequency by Gender Across Languages')
plt.xlabel('Language')
plt.ylabel('Function Word Frequency (Mean)')
plt.legend(title='Gender', loc='best')
plt.tight_layout()
plot2_path = os.path.join(results_dir, "gender_writing_metrics.png")
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Gender-Based Writing Metrics plot saved to '{plot2_path}'")