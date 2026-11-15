import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier 

# Step 1 Loading the dataset

df = pd.read_csv('iris.csv')


print('--- First 5 rows of the dataset ---')
print(df.head())
print('\n')


# Step 2: Exploratory Data Analysis(Pairplot)

# Visualize feature relationships grouped by species using a seaborn pairplot

sns.pairplot(df, hue='species', diag_kind='kde')
plt.suptitle('Pairplot of Iris Dataset by species', y=1.02)
plt.show()

# Step 3:Prepare Data and train or test split
# Separate features (X) from the target label (y)

X = df.drop(columns=['species'])  # Input features
y = df['species']  # Target label(species)

# Split data into 80% training set and 20% testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Initialize the StandardScaler and normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train and evaluate Logistic Regression

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Predict test set results
y_pred_log = log_reg.predict(X_test_scaled)

print('--- Logistic Regression Performance ---')
print(classification_report(y_test, y_pred_log))

# Step 5: Train and evaluate Decision Tree
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

y_pred_tree = tree_clf.predict(X_test)

print('--- Decision Tree Classifier Performance ---')
print(classification_report(y_test, y_pred_tree))


# Step6: Plot Confusion Matrix Heatmaps

# Setting up  figure with two side-by-side subplots for our confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))


# 1. Confusion Matrix for Logistic Regression
cm_log = confusion_matrix(y_test, y_pred_log)
sns.heatmap(
    cm_log,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[0],
    xticklabels=log_reg.classes_,
    yticklabels=log_reg.classes_,
)

axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# 2. Confusion Matrix for Decision Tree

cm_tree = confusion_matrix(y_test, y_pred_tree)
sns.heatmap(
    cm_tree,
    annot=True,
    fmt='d',
    cmap='Greens',
    ax=axes[1],
    xticklabels=tree_clf.classes_,
    yticklabels=tree_clf.classes_,
)

axes[1].set_title('Decision Tree Confusion Matrix')
axes[1].set_xlabel('Predicting Label')
axes[1].set_ylabel('True Label')

# Adjusting layout and display the plots
plt.tight_layout()
plt.show()
