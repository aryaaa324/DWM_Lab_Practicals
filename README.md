# DWM Practical Aims:
1. To Study Data mining tool Weka and create new ARFF File.
2. To perform treatment of Missing Values of the attributes in Weka.
3. To understand and implement OLAP operations on a multi-dimensional data cube.
4. To treat missing values with different techniques in Python
5. To perform EDA on the given dataset
6. To implement various data flow transformations that are commonly used in ETL Processes
7. To implement Apriori Algorithm
8. To implement Naive Bayes Algorithm
9. To implement KNN Algorithm
10. To implement K-means Algorithm
11. To implement Decision Tree Algorithm
12. To implement Linear Regression Algorithm
---
## 1. 📁 Weka & ARFF File Creation
### Steps:
Open Weka GUI.
-> Choose Explorer → Click on Preprocess tab.
-> Use the "Open File" button to load a dataset or "Edit" to create a new one.
-> Save your dataset in ARFF format (.arff).

Understand the ARFF structure:
@relation – dataset name.
@attribute – features and their types.
@data – actual data records.

---

## 2. 🧩 Handling Missing Values in Weka
### Steps:
-> Load the dataset with missing values in Weka.
-> Navigate to the Preprocess tab.
-> Use Filters > unsupervised > attribute > ReplaceMissingValues.
-> Apply the filter to replace missing data using default strategies (mean/mode).
-> Save the clean dataset.

---

## 3. 🧊 OLAP Operations
### Key Operations:
Roll-up: Aggregating data by climbing a concept hierarchy.
Drill-down: Breaking data into finer levels.
Slice: Selecting a single dimension.
Dice: Selecting two or more dimensions.
Pivot (Rotate): Reorienting the data cube view.

Example:
Cube: Sales data → Dimensions: Time, Product, Region
Roll-up: Product level → Category
Slice: Only 2023 data
Dice: Only Electronics in Q1

## 4. 🧼 Handling Missing Values in Python
### Techniques:
-> Replacing missing values with mean, median and mode by computing strategy=mean/median/most_frequent

Other Techniques: 
Removing rows/columns – df.dropna()
Replacing with mean/median/mode
df.fillna(df.mean())
Forward/Backward fill – df.fillna(method='ffill')

---

## 5. 📊 Exploratory Data Analysis (EDA)
### Steps:

->Load dataset using pandas.
->Understand structure – df.info(), df.describe()
->Univariate analysis – Histograms, Countplots
->Bivariate analysis – Scatter plots, Correlation matrix
->Check for outliers – Boxplots
->Handle skewness, missing values
->Visualize distributions using matplotlib, seaborn

## 6. 🔄 ETL Data Flow Transformations
### Common Transformations:
Filtering – Removing unwanted rows/columns.
Aggregation – Summing, counting, averaging.
Join/Merge – Combining multiple data sources.
Sorting & Ordering – For cleaner visual flow.
Data Type Conversion – e.g., string to datetime.
Standardization/Normalization – For scaling features.

## 7. 🛒 Apriori Algorithm
### Steps:
->Set minimum support and confidence thresholds.
->Generate frequent itemsets:
->Start with single items → prune low-support.
->Combine to form pairs, triplets...
->Generate association rules:
->For each frequent itemset, create rules.
->Retain rules that meet confidence threshold.

## 8. 🎯 Naive Bayes Algorithm
### Steps:
->Calculate prior probability for each class.
->For each feature, calculate likelihood: P(feature | class)
-> Apply Bayes Theorem to compute posterior: P(class | features) ∝ P(class) * P(features | class)
-> Assign class with the highest posterior.

## 9. 🧍‍♂️ K-Nearest Neighbors (KNN)
### Steps:
->Choose K (number of neighbors).
->Calculate distance (e.g., Euclidean) from the test point to all training points.
->Select K nearest neighbors.
->For classification: majority vote
->For regression: average of K neighbors.
->Predict the result accordingly.

## 10. 🔵 K-Means Clustering
### Steps:
->Choose number of clusters K.
->Initialize K centroids randomly.
->Assign each point to the nearest centroid.
->Update centroid: mean of points in each cluster.
->Repeat until centroids stabilize (convergence).

## 11. 🌳 Decision Tree Algorithm
### Steps:
->Start with the entire dataset.
->Select the best feature using:
->Information Gain or Gini Index
->Split the dataset based on feature values.
->Repeat the process recursively on subsets.

Stop when:
->All records belong to one class, or
->Max depth reached, or
->No gain possible.

## 12. 📈 Linear Regression
### Steps:
-> Assume a linear relationship: Y = mX + c
->Estimate coefficients m and c by minimizing the cost function (MSE).
->Use Gradient Descent or Normal Equation for optimization.
->Predict using the trained model:
->Input feature → Output continuous value.
