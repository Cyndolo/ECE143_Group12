# ECE143_Group12

We are using the folliwing third party modules:
- Numpy
- Pandas
- Matplotlib
- Seaborn
- sklearn
- xgboost
- os
- statsmodels.api
- scipy.stats

There are 5 separate Python files that each produce different charts:
1) yuxin_code.py
2) MBA_analysis_Henson_Le.py
3) haochen.py
4) Zheng_code.py
5) .

yuxin_code.py:
- Handling missing values, encoding categorical variables, standardizing numerical features, and splitting into training and testing sets.
- Various classification models, including Random Forest, XGBoost, Decision Tree, KNN, and others, are trained to predict MBA admissions, with accuracy scores used for evaluation.
- Key features are selected using Random Forest importance, and models are retrained with the top five features for improved efficiency.
- A stacking ensemble combines multiple models, using Logistic Regression as the meta-classifier for enhanced predictive performance.
- Regression models analyze relationships between GPA, GMAT, work experience, and admission probability, evaluated with MSE and R² scores.

MBA_analysis_Henson_Le.py:
- This file generates three spider/radar charts displaying the distribution of work industries each major in the dataset.
- The file also generates a heatmap showing combinations of majors and work industries and their corresponding admittance rates.
- All you have to do is make sure the filename called by "file_path" matches your file name.
- Click run and your charts should be generated!

haochen.py:
- Generates Violin plots for gender-conditional distributions of GPA, GMAT and work experience
- Generates Violin plots for race-conditional distributions of GPA, GMAT and work experience
- Generates histograms for gender-conditional admission counts and rates
- Generates histograms for race-conditional admission counts and rates
- Generates GPA vs. GMAT linear model plot, and its associated residal plot and GMAT-conditional GPA variance plot.

Zheng_code.py:
- Generates count plot for admission statistics
- Generates the count plot of working experience for the applicants in the top 4 working industries: Nonprofit, Consulting, Technology and PE/VC.
- Groups applicants by working experience and plots working experience against admission rate. Note that samples with working experience less than 2 years or more than 8 years are droped as the sample numbers are too small.
- Modify the filename in the read_csv function accordingly to run the code successfully.

  

