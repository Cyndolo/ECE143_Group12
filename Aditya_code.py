import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Set up styling
sns.set(style="whitegrid", palette="pastel", font="Arial", rc={"axes.titlesize": 18, "axes.labelsize": 12})
plt.rcParams['figure.dpi'] = 300

# Load data
file_path = "MBA.csv"  # Replace with actual file path
df = pd.read_csv(file_path)

# Data Cleaning
df['work_industry'] = df['work_industry'].str.strip().fillna('Unknown')

# Count occurrences of each industry
industry_counts_df = df['work_industry'].value_counts().reset_index()
industry_counts_df.columns = ['Industry', 'Count']

# Visualization - Industry Bar Plot
plt.figure(figsize=(14, 10))  # Increase figure size
industry_plot = sns.barplot(
    data=industry_counts_df.sort_values('Count', ascending=False),
    y='Industry',
    x='Count',
    orient='h',
    edgecolor='black',
    linewidth=1,
    color='lightblue'
)

plt.title('MBA Applicants by Work Industry', pad=20, fontsize=18, weight='bold', loc='center')
plt.xlabel('Number of Applicants', labelpad=15, fontsize=14)
plt.ylabel('Industry', labelpad=15, fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=8)  # Reduce industry label font size
plt.xlim(0, industry_counts_df['Count'].max() + 20)

# Add value labels
for p in industry_plot.patches:
    width = p.get_width()
    plt.text(width + 5, 
             p.get_y() + p.get_height()/2, 
             f'{int(width)}', 
             ha='left', 
             va='center',
             fontsize=12)

plt.tight_layout()
plt.show()
