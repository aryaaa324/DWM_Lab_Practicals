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

💡 **Goal**: Learn how to define datasets in `.arff` (Attribute-Relation File Format).

### 🧭 Steps:
1. Open **Weka GUI**  
2. Choose **Explorer** → Click on **Preprocess** tab  
3. Use the **"Open File"** button to load a dataset or **"Edit"** to create a new one  
4. Save your dataset in **ARFF format (.arff)**  

### 🏗️ ARFF Structure:
- `@relation` – dataset name  
- `@attribute` – features and their types  
- `@data` – actual data records  

---

## 2. 🧩 Handling Missing Values in Weka  
**Definition**: Missing values occur when no data value is stored for a variable in an observation.

🎯 **Goal**: Replace missing values using Weka's filters.

### 🧭 Steps:
1. Load the dataset with missing values in Weka  
2. Navigate to the **Preprocess** tab  
3. Use **Filters > unsupervised > attribute > ReplaceMissingValues**  
4. Apply the filter to replace missing data using default strategies (mean/mode)  
5. Save the cleaned dataset  

---

## 3. 🧊 OLAP Operations  
**Definition**: OLAP (Online Analytical Processing) provides multidimensional analysis of business data and supports complex queries.

### 🛠️ Key Operations:
- **Roll-up**: Aggregating data by climbing a concept hierarchy  
- **Drill-down**: Breaking data into finer levels  
- **Slice**: Selecting a single dimension  
- **Dice**: Selecting two or more dimensions  
- **Pivot (Rotate)**: Reorienting the data cube view  

📦 **Example**:  
Cube: Sales data → Dimensions: Time, Product, Region  
- Roll-up: Product level → Category  
- Slice: Only 2023 data  
- Dice: Only Electronics in Q1  

---

## 4. 🧼 Handling Missing Values in Python  
**Definition**: Python offers flexible techniques to clean and fill missing values programmatically.

### 🔧 Techniques:
- Replace missing values using **mean**, **median**, or **mode**  
  - `df.fillna(df.mean())`, `df.fillna(df.median())`  
- **Remove rows/columns**: `df.dropna()`  
- **Forward/Backward Fill**: `df.fillna(method='ffill')`, `df.fillna(method='bfill')`  
- **Custom imputation** based on domain knowledge  

---

## 5. 📊 Exploratory Data Analysis (EDA)  
**Definition**: EDA is the process of examining datasets visually and statistically to discover patterns, spot anomalies, and test assumptions.

### 🧭 Steps:
1. Load dataset using `pandas`  
2. Understand structure – `df.info()`, `df.describe()`  
3. Univariate analysis – **Histograms**, **Countplots**  
4. Bivariate analysis – **Scatter plots**, **Correlation matrix**  
5. Check for outliers – **Boxplots**  
6. Handle skewness and missing values  
7. Visualize data using `matplotlib`, `seaborn`  

---

## 6. 🔄 ETL Data Flow Transformations  
**Definition**: ETL stands for **Extract, Transform, Load**—a process used in data warehousing to prepare and integrate data.

### 🔧 Common Transformations:
- **Filtering** – Removing unnecessary rows/columns  
- **Aggregation** – Summing, counting, averaging  
- **Join/Merge** – Combining multiple datasets  
- **Sorting & Ordering** – For clean visual output  
- **Data Type Conversion** – e.g., string to datetime  
- **Standardization/Normalization** – For feature scaling  

---

## 7. 🛒 Apriori Algorithm  
**Definition**: An algorithm for mining frequent itemsets and generating association rules from transactional data.

### 🧭 Steps:
1. Set **minimum support** and **confidence** thresholds  
2. Generate frequent itemsets:  
   - Start with single items  
   - Prune low-support itemsets  
   - Combine to form pairs, triplets...  
3. Generate association rules:  
   - Create rules from frequent itemsets  
   - Retain those meeting the confidence threshold  

---

## 8. 🎯 Naive Bayes Algorithm  
**Definition**: A probabilistic classifier based on **Bayes' Theorem**, assuming independence among features.

### 🧭 Steps:
1. Calculate **prior probability** for each class  
2. For each feature, compute **likelihood**: `P(feature | class)`  
3. Apply **Bayes Theorem** to compute posterior:  
   `P(class | features) ∝ P(class) * P(features | class)`  
4. Predict class with **highest posterior probability**  

---

## 9. 🧍‍♂️ K-Nearest Neighbors (KNN)  
**Definition**: A non-parametric method used for classification and regression based on proximity.

### 🧭 Steps:
1. Choose value of **K** (number of neighbors)  
2. Calculate **distance** (e.g., Euclidean) from the test point to all training points  
3. Select the **K nearest neighbors**  
4. For classification: use **majority vote**  
5. For regression: take **average of K neighbors**  
6. Predict the result accordingly  

---

## 10. 🔵 K-Means Clustering  
**Definition**: An unsupervised learning algorithm that partitions data into **K distinct clusters**.

### 🧭 Steps:
1. Choose number of clusters **K**  
2. Initialize **K centroids randomly**  
3. Assign each point to the **nearest centroid**  
4. Update centroids as the **mean of points in each cluster**  
5. Repeat steps 3–4 until **centroids converge**  

---

## 11. 🌳 Decision Tree Algorithm  
**Definition**: A tree-like model used for classification and decision-making based on features.

### 🧭 Steps:
1. Start with the **entire dataset**  
2. Select the **best feature** to split using  
   - **Information Gain** or  
   - **Gini Index**  
3. Split the dataset based on feature values  
4. Repeat recursively for each child node  

📌 **Stopping Criteria**:
- All records belong to one class  
- No further information gain  
- Maximum depth is reached  

---

## 12. 📈 Linear Regression  
**Definition**: A supervised learning algorithm to model the relationship between one or more independent variables and a dependent variable.

### 🧭 Steps:
1. Assume a linear model: `Y = mX + c`  
2. Estimate coefficients **m** and **c** by minimizing **Mean Squared Error (MSE)**  
3. Use **Gradient Descent** or **Normal Equation** for optimization  
4. Predict using the trained model to output continuous values  

---
