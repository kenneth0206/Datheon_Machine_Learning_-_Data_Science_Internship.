# Datheon_Machine_Learning_-_Data_Science_Internship.


#Multi-Class Benchmark Classisfication Pipeline(Iris Dataset)

# Project Overview and Steps
1)Data Loading(pandas): Loads the iris.csv dataset and displays the initial rows
2)Exploratory Data Analysis(seaborn and matplotlib): Generates a pairplot showing feature distributions and relationships grouped by flower species.
3) Data Preparation and splitting(scikit-learn):
   -Separates input features (X) from target labels (y).
   -Splits data into 80% training and 20% testing sets using train_test_split.
   -Normalizes feature values using StandardScaler to prevent feature dominance.
4) Logistic Regression: Trains a linear classifier on scaled training data and evaluates perfomance via precision, recall, and F1-score.
5)Decision Tree Classifier: Trains a non-linear tree-based classifier and evaluates perfomance.
6)Confusion Matrices:Plots side-by-side heatmaps comparing true labels versus predicted labels for both models.

# Packages
-Pandas: For loading, managing dataset
-numpy: for foundational array and numerical operations
-matplotlib: for configuring figures, subplots, and displaying visual plots
-seaborn:For stylized statsistical data visualisations for an example heatmaps and pairplots.
-scikit-learn(sklearn): For data processing 
