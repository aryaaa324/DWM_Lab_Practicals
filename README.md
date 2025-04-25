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
## 🔢 Practical Aims Overview

| 🔢 No. | 🧪 Practical Aim                  | 📝 Description                                                            |
|-------|----------------------------------|---------------------------------------------------------------------------|
| 1️⃣    | Study Weka & Create ARFF         | Understand the Weka tool and how to create `.arff` files used for data mining tasks. |
| 2️⃣    | Missing Values in Weka           | Learn how to treat missing data using Weka’s built-in filters and preprocessing tools. |
| 3️⃣    | OLAP Operations                  | Explore Roll-up, Drill-down, Slice, Dice, and Pivot on multidimensional data cubes. |
| 4️⃣    | Handle Missing Values in Python  | Use Python to treat missing values using various imputation techniques.   |
| 5️⃣    | Exploratory Data Analysis (EDA)  | Visualize and summarize datasets using Python's EDA techniques.           |
| 6️⃣    | ETL Data Flow Transformations    | Implement Extract, Transform, Load (ETL) concepts using basic transformation steps. |
| 7️⃣    | Apriori Algorithm                | Discover frequent itemsets and generate association rules.                |
| 8️⃣    | Naive Bayes Algorithm            | Build a probabilistic classifier based on Bayes' Theorem.                 |
| 9️⃣    | K-Nearest Neighbors              | Classify data points based on their similarity to neighbors.              |
| 🔟     | K-Means Clustering               | Group data points into K distinct clusters.                               |
| 1️⃣1️⃣  | Decision Tree Algorithm          | Classify data by learning decision rules from features.                   |
| 1️⃣2️⃣  | Linear Regression                | Predict continuous outcomes based on linear relationships.                |

---

## 1. 📁 Weka & ARFF File Creation
**Definition**: Weka is a GUI-based data mining tool that supports tasks such as preprocessing, classification, regression, clustering, and visualization.

💡 Goal: Learn how to define datasets in .arff (Attribute-Relation File Format).

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
**Definition** : Missing values occur when no data value is stored for a variable in an observation.

🎯 Goal: Replace missing values using Weka's filters like:
->Mean/Mode replacement
->Custom value assignment

### Steps:
-> Load the dataset with missing values in Weka.
-> Navigate to the Preprocess tab.
-> Use Filters > unsupervised > attribute > ReplaceMissingValues.
-> Apply the filter to replace missing data using default strategies (mean/mode).
-> Save the clean dataset.

---

## 3. 🧊 OLAP Operations
**Definition**: OLAP (Online Analytical Processing) provides multidimensional analysis of business data and supports complex queries.
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
**Definition**: Python offers flexible techniques to clean and fill missing values programmatically.
### Techniques:
-> Replacing missing values with mean, median and mode by computing strategy=mean/median/most_frequent

Other Techniques: 
Removing rows/columns – df.dropna()
Replacing with mean/median/mode
df.fillna(df.mean())
Forward/Backward fill – df.fillna(method='ffill')

---

## 5. 📊 Exploratory Data Analysis (EDA)
**Definition**: EDA is the process of examining datasets visually and statistically to discover patterns, spot anomalies, and test assumptions.

### Steps:

->Load dataset using pandas.
->Understand structure – df.info(), df.describe()
->Univariate analysis – Histograms, Countplots
->Bivariate analysis – Scatter plots, Correlation matrix
->Check for outliers – Boxplots
->Handle skewness, missing values
->Visualize distributions using matplotlib, seaborn

---
## 6. 🔄 ETL Data Flow Transformations
**Definition**: ETL stands for Extract, Transform, Load—a process used in data warehousing to prepare data.

### Common Transformations:
Filtering – Removing unwanted rows/columns.
Aggregation – Summing, counting, averaging.
Join/Merge – Combining multiple data sources.
Sorting & Ordering – For cleaner visual flow.
Data Type Conversion – e.g., string to datetime.
Standardization/Normalization – For scaling features.

---
## 7. 🛒 Apriori Algorithm
**Definition**: An algorithm for mining frequent itemsets and learning association rules.

### Steps:
->Set minimum support and confidence thresholds.
->Generate frequent itemsets:
->Start with single items → prune low-support.
->Combine to form pairs, triplets...
->Generate association rules:
->For each frequent itemset, create rules.
->Retain rules that meet confidence threshold.

---
## 8. 🎯 Naive Bayes Algorithm
**Definition**: A probabilistic classifier based on Bayes' Theorem assuming feature independence.

### Steps:
->Calculate prior probability for each class.
->For each feature, calculate likelihood: P(feature | class)
-> Apply Bayes Theorem to compute posterior: P(class | features) ∝ P(class) * P(features | class)
-> Assign class with the highest posterior.

---
## 9. 🧍‍♂️ K-Nearest Neighbors (KNN)
**Definition**: A non-parametric method used for classification and regression.

### Steps:
->Choose K (number of neighbors).
->Calculate distance (e.g., Euclidean) from the test point to all training points.
->Select K nearest neighbors.
->For classification: majority vote
->For regression: average of K neighbors.
->Predict the result accordingly.

---
## 10. 🔵 K-Means Clustering
**Definition**: An unsupervised algorithm that partitions data into K clusters.

### Steps:
->Choose number of clusters K.
->Initialize K centroids randomly.
->Assign each point to the nearest centroid.
->Update centroid: mean of points in each cluster.
->Repeat until centroids stabilize (convergence).

---
## 11. 🌳 Decision Tree Algorithm
**Definition**: A tree-like model used for decision-making and classification.

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

---
## 12. 📈 Linear Regression
**Definition**: A method to model the relationship between a dependent and one/more independent variables.

### Steps:
-> Assume a linear relationship: Y = mX + c
->Estimate coefficients m and c by minimizing the cost function (MSE).
->Use Gradient Descent or Normal Equation for optimization.
->Predict using the trained model:
->Input feature → Output continuous value.
