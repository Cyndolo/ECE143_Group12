import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "MBA.csv"
df = pd.read_csv(file_path)

df.info(), df.head()
#%%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['gpa'], bins=20, kde=True, ax=axes[0])
axes[0].set_title("GPA Distribution")
sns.histplot(df['gmat'], bins=20, kde=True, ax=axes[1])
axes[1].set_title("GMAT Distribution")
sns.histplot(df['work_exp'], bins=20, kde=True, ax=axes[2])
axes[2].set_title("Work Experience (Years) Distribution")
plt.show()
#%%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
df['gender'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[0])
axes[0].set_title("Gender Distribution")
df['international'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1])
axes[1].set_title("International Student Ratio")
plt.show()
#%%
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
df['major'].value_counts().plot(kind='bar', ax=axes[0])
axes[0].set_title("Number of Applicants by Major")
df['work_industry'].value_counts().plot(kind='bar', ax=axes[1])
axes[1].set_title("Number of Applicants by Work Industry")
plt.xticks(rotation=45)
plt.show()
#%%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(x="gender", y="gpa", data=df, ax=axes[0])
axes[0].set_title("GPA Distribution by Gender")
sns.boxplot(x="gender", y="gmat", data=df, ax=axes[1])
axes[1].set_title("GMAT Distribution by Gender")
plt.show()
#%%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(x="international", y="gpa", data=df, ax=axes[0])
axes[0].set_title("GPA: International vs Domestic Students")
sns.boxplot(x="international", y="gmat", data=df, ax=axes[1])
axes[1].set_title("GMAT: International vs Domestic Students")
plt.show()
#%%
admission_rate = df.groupby("work_industry")["admission"].apply(lambda x: x.notnull().mean()).sort_values()
plt.figure(figsize=(12, 6))
admission_rate.plot(kind="bar")
plt.title("Admission Rate by Work Industry")
plt.ylabel("Admission Rate")
plt.xticks(rotation=45)
plt.show()
#%%
sns.pairplot(df[['gpa', 'gmat', 'work_exp']])
plt.show()
#%%
plt.figure(figsize=(8, 6))
sns.heatmap(df[['gpa', 'gmat', 'work_exp']].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of GPA, GMAT, and Work Experience")
plt.show()
#%%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
#%%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

file_path = "MBA.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path)

print(df.info())
print(df.head())

df['admission'] = df['admission'].apply(lambda x: 1 if x == "Admit" else 0)

categorical_cols = ['gender', 'international', 'major', 'race', 'work_industry']
existing_categorical_cols = [col for col in categorical_cols if col in df.columns]

label_encoders = {}
for col in existing_categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['application_id', 'admission'], errors='ignore')
y = df['admission']

numerical_cols = ['gpa', 'gmat', 'work_exp']
existing_numerical_cols = [col for col in numerical_cols if col in X.columns]

if existing_numerical_cols:
    scaler = StandardScaler()
    X[existing_numerical_cols] = scaler.fit_transform(X[existing_numerical_cols].copy())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.head())
print(y_train.head())
#%%
if X_train.empty or X_test.empty:
    raise ValueError("Training or testing dataset is empty!")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# Feature Importance Plot
rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=rf_importance.values, y=rf_importance.index)
plt.title("Random Forest Feature Importance")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.show()

print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
#%%

xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

xgb_importance = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=xgb_importance.values, y=xgb_importance.index)
plt.title("XGBoost Feature Importance")
plt.show()

print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

#%%
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

dt_importance = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=dt_importance.values, y=dt_importance.index)
plt.title("Decision Tree Feature Importance")
plt.show()
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
#%%
top_features = rf_importance.index[:5]
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]
#%%
top_rf = RandomForestClassifier(n_estimators=100, random_state=42)
top_rf.fit(X_train_selected, y_train)
y_pred_top_rf = top_rf.predict(X_test_selected)
top_rf_accuracy = accuracy_score(y_test, y_pred_top_rf)

top_xgb = XGBClassifier(eval_metric='logloss')
top_xgb.fit(X_train_selected, y_train)
y_pred_top_xgb = top_xgb.predict(X_test_selected)
top_xgb_accuracy = accuracy_score(y_test, y_pred_top_xgb)

top_dt = DecisionTreeClassifier(random_state=42)
top_dt.fit(X_train_selected, y_train)
y_pred_top_dt = top_dt.predict(X_test_selected)
top_dt_accuracy = accuracy_score(y_test, y_pred_top_dt)

print(f"Random Forest (Top Features) Accuracy: {top_rf_accuracy:.4f}")
print(f"XGBoost (Top Features) Accuracy: {top_xgb_accuracy:.4f}")
print(f"Decision Tree (Top Features) Accuracy: {top_dt_accuracy:.4f}")
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier

#%%
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Accuracy: {log_reg_accuracy:.4f}")
#%%
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {svm_accuracy:.4f}")
#%%
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
lgbm_accuracy = accuracy_score(y_test, y_pred_lgbm)
print(f"LightGBM Accuracy: {lgbm_accuracy:.4f}")
#%%
mlp = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
print(f"MLP Neural Network Accuracy: {mlp_accuracy:.4f}")
#%%
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('svm', svm),
        ('lgbm', lgbm)
    ],
    final_estimator=LogisticRegression()
)
stacking_clf.fit(X_train, y_train)
y_pred_stacking = stacking_clf.predict(X_test)
stacking_accuracy = accuracy_score(y_test, y_pred_stacking)
print(f"Stacking Ensemble Accuracy: {stacking_accuracy:.4f}")
#%%
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
import pandas as pd
import numpy as np

models = {
    'Linear_Regression': LinearRegression(),
    'KNN': KNeighborsRegressor(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(),
    'XGB': XGBRegressor(),
    'SVM': SVR()  # SVM回归
}

is_classification = len(np.unique(y_train)) <= 10

results = {'Model': []}

if is_classification:
    results.update({'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []})
else:
    results.update({'MSE': [], 'R2 Score': []})

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results['Model'].append(name)

    if is_classification:
        pred_binary = (predictions > 0.5).astype(int)
        results['Accuracy'].append(accuracy_score(y_test, pred_binary))
        results['Precision'].append(precision_score(y_test, pred_binary, average='weighted', zero_division=0))
        results['Recall'].append(recall_score(y_test, pred_binary, average='weighted', zero_division=0))
        results['F1-Score'].append(f1_score(y_test, pred_binary, average='weighted'))
    else:
        results['MSE'].append(mean_squared_error(y_test, predictions))
        results['R2 Score'].append(r2_score(y_test, predictions))

results_df = pd.DataFrame(results)
print(results_df)