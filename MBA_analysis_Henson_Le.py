import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure the .csv file is in your immediate directory
# Otherwise, change the file_path to point towards it 

file_path = "MBA.csv"
df = pd.read_csv(file_path)

# The "admission" column contained empty cells
# We will fill them in with "Reject"

df['admission'] = df['admission'].fillna('Reject')

# This section will create three radar/spider charts displaying the distribution of work industries for each major

admission_counts = df.groupby(["major", "work_industry", "admission"]).size().unstack(fill_value=0)

admission_counts["Acceptance Rate"] = admission_counts["Admit"] / (admission_counts["Admit"] + admission_counts["Reject"])

industry_major_counts = df.groupby(["work_industry", "major"]).size().unstack(fill_value=0)

# Rename certain work industry labels to improve readability
industry_major_counts = industry_major_counts.rename(index={
    "Investment Management": "Management",
    "Financial Services": "Finance",
    "Investment Banking": "Banking"
})

industry_totals = industry_major_counts.sum(axis=1).sort_values(ascending=False)
sorted_industries = industry_totals.index.tolist()

industry_major_counts = industry_major_counts.loc[sorted_industries]

major_colors = {"Humanities": "red", "Business": "blue", "STEM": "green"}

num_industries = len(sorted_industries)
angles = np.linspace(0, 2 * np.pi, num_industries, endpoint=False).tolist()

for major in industry_major_counts.columns:
    values = industry_major_counts[major].tolist()
    
    angles = np.linspace(0, 2 * np.pi, num_industries, endpoint=False).tolist()

    values.append(values[0])
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=150)
    
    ax.plot(angles, values, label=major, linewidth=1.5, color=major_colors.get(major, "black"))
    ax.fill(angles, values, alpha=0.2, color=major_colors.get(major, "black"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(sorted_industries, fontsize=10, rotation=0, ha='center', va='top')
    ax.set_title(f"{major} - Work Industry Distribution", fontsize=14, pad=20)
    
    # Adjust label positions so that words do not overlap with the chart
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        label.set_y(label.get_position()[1] - 0.08)  # Move labels further from the chart

    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    plt.show()


# This section will create a heatmap displaying the acceptance rate of every major and work industry combination

sorted_work_industries = admission_counts.groupby("work_industry")["Acceptance Rate"].mean().sort_values(ascending=True).index.tolist()
acceptance_matrix = admission_counts["Acceptance Rate"].unstack().reindex(columns=sorted_work_industries)

plt.figure(figsize=(20, 8))
sns.heatmap(acceptance_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Acceptance Rates by Major and Work Industry")
plt.xlabel("Work Industry")
plt.ylabel("Major")
plt.xticks(rotation=45, ha='right')
plt.show()