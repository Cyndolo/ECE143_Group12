import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "MBA.csv"
df = pd.read_csv(file_path)
df['admission'] = df['admission'].fillna('Reject')
df['race'] = df['race'].fillna('Other')
df

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from scipy.stats import pearsonr

x = df['gmat']
y = df['gpa']

# Assume x and y are given
X = sm.add_constant(x)  # Add intercept for regression
model = sm.OLS(y, X).fit()  # Ordinary Least Squares Regression
y_pred = model.predict(X)  # Predicted values
residuals = y - y_pred

r, _ = pearsonr(x, y)
intercept, slope = model.params[0], model.params[1]
print(f"Regression line: y = {intercept:.3f} + {slope:.3f}x")
print(f"Pearson Correlation: r = {r:.3f}")

# Estimate conditional variance: Bin X and compute residual variance in each bin
bin_edges = np.linspace(min(x), max(x), 21)  # Define bins
bin_indices = np.digitize(x, bins=bin_edges)
conditional_var = [np.var(residuals[bin_indices == i]) for i in range(1, len(bin_edges))]

# Plot conditional variance
plt.figure(figsize=(12, 4))

# Subplot 1: Conditional Variance Plot
plt.subplot(1, 3, 3)
plt.plot(bin_edges[:-1], conditional_var, marker='o', linestyle='--')
plt.xlabel("GMAT")
plt.ylabel("Conditional Variance of GPA | GMAT")
plt.title("Conditional Variance")

# Subplot 2: 2D Scatter Plot and Fitted Line
plt.subplot(1, 3, 1)
plt.scatter(x, y, label="Data", color="blue", alpha=0.5)
plt.plot(x, y_pred, label="Fitted OLS Line", color="red", linewidth=2)
plt.xlabel("GMAT")
plt.ylabel("GPA")
plt.title("GPA vs GMAT")
plt.legend()

# Subplot 3: Residuals Plot
plt.subplot(1, 3, 2)
plt.scatter(x, residuals, label="Residuals", color="green", alpha=0.5)
plt.axhline(y=0, color='red', linewidth=2)
plt.xlabel("GMAT")
plt.title("Residuals Plot")

plt.tight_layout()
plt.show()

pd.crosstab([df['gender'], df['race'].fillna('Other')], df['admission'], margins=True)

plt.rcParams.update({'font.size': 15})

# Set up the figure for two subplots (for GPA and GMAT)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.violinplot(x='race', y='gpa', data=df, ax=axes[0], palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
axes[0].set_title('GPA Distribution by Race')
sns.violinplot(x='race', y='gmat', data=df, ax=axes[1], palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
axes[1].set_title('GMAT Distribution by Race')
sns.violinplot(x='race', y='work_exp', data=df, ax=axes[2], palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
axes[2].set_title('work experience Distribution by Race')
plt.xticks(fontsize=14)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.violinplot(x='gender', y='gpa', data=df, ax=axes[0], palette={"Male": "cyan", "Female": "red"})
axes[0].set_title('GPA Distribution by Gender')
sns.violinplot(x='gender', y='gmat', data=df, ax=axes[1], palette={"Male": "cyan", "Female": "red"})
axes[1].set_title('GMAT Distribution by Gender')
sns.violinplot(x='gender', y='work_exp', data=df, ax=axes[2], palette={"Male": "cyan", "Female": "red"})
axes[2].set_title('work experience Distribution by Gender')
plt.xticks(fontsize=14)
plt.tight_layout()
plt.show()

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# sns.boxplot(x='gender', y='gpa', data=df, ax=axes[0])
# axes[0].set_title('GPA Distribution by Gender')
# sns.boxplot(x='gender', y='gmat', data=df, ax=axes[1])
# axes[1].set_title('GMAT Distribution by Gender')
# sns.boxplot(x='gender', y='work_exp', data=df, ax=axes[2])
# axes[2].set_title('work experience Distribution by Gender')
# plt.xticks(fontsize=14)
# plt.tight_layout()
# plt.show()

plt.rcParams.update({'font.size': 13})
ax = sns.countplot(x='race', hue='admission', data=df)

# Add percentage labels to the bars
races = ax.get_xticklabels()
sorted_patches = sorted(ax.patches[:15], key=lambda patch: patch.get_x())
for idx,p in enumerate(sorted_patches):
    height = p.get_height()
    race = races[idx//3].get_text()
    race_total = df[df['race'] == race].shape[0]
    percentage = (height / race_total) * 100  # Calculate the percentage
    ax.text(p.get_x() + p.get_width() / 2, height + 1, f'{percentage:.1f}%', 
            ha='center', va='bottom', fontsize=10)  # Add the text label
plt.show()

plt.rcParams.update({'font.size': 13})
ax = sns.countplot(x='gender', hue='admission', data=df)

# Add percentage labels to the bars
genders = ax.get_xticklabels()
sorted_patches = sorted(ax.patches[:6], key=lambda patch: patch.get_x())
for idx,p in enumerate(sorted_patches):
    height = p.get_height()
    gender = genders[idx//3].get_text()
    gender_total = df[df['gender'] == gender].shape[0]
    percentage = (height / gender_total) * 100  # Calculate the percentage
    ax.text(p.get_x() + p.get_width() / 2, height + 1, f'{percentage:.1f}%', 
            ha='center', va='bottom', fontsize=10)  # Add the text label
plt.show()

