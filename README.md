# Customer Segmentation with K-Means Clustering

An exploratory machine learning project that uses Python, Exploratory Data Analysis (EDA), and K-Means clustering to identify customer segments based on annual income and spending behaviour.

**Repository:** [Customer Segmentation with K-Means](https://github.com/vinayak932/Customer_Segmentation_with_K-mean)

## Overview

Businesses often have customers with very different spending behaviours, even when their incomes are similar. Treating every customer as part of the same group can make it difficult to understand these differences.

In this project, I used unsupervised machine learning to discover customer segments from a mall customer dataset.

The project focuses on understanding the data before applying a model. It includes data inspection, exploratory data analysis, distribution analysis, outlier detection, correlation analysis, feature scaling, and K-Means clustering.

The final model identifies five customer segments with distinct income and spending characteristics.

## Objectives

- Perform exploratory data analysis using Pandas and visualisation libraries.
- Understand numerical distributions, skewness, and potential outliers.
- Investigate relationships between age, annual income, and spending score.
- Compare customer spending patterns across genders.
- Understand why feature scaling is important for distance-based algorithms.
- Apply K-Means clustering to customer data.
- Use the Elbow Method and Silhouette Score to evaluate different cluster counts.
- Interpret the resulting clusters and translate them into understandable customer profiles.

## Dataset

**Dataset:** [Mall Customer Segmentation Data on Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

The dataset contains 200 customer records and five columns.

| Feature | Description |
|---|---|
| CustomerID | Unique identifier for each customer |
| Gender | Customer gender |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands of dollars |
| Spending Score (1-100) | A score assigned by the mall based on customer behaviour |

The spending score is a mall-assigned measure, not the actual amount spent in dollars.

## Tech Stack

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualisation
- **Seaborn** — Statistical visualisation
- **Scikit-learn** — Feature scaling and K-Means clustering
- **Jupyter Notebook** — Interactive analysis

## Project Workflow

### 1. Data Inspection

I started by examining the dataset's structure, data types, descriptive statistics, and missing values.

The dataset contains 200 rows and five columns. No missing values were found during the initial inspection.

I also examined the descriptive statistics to understand the central tendency and spread of the numerical features.

### 2. Exploratory Data Analysis

EDA was used to understand customer characteristics before training the model.

The analysis included:

- Histograms to examine numerical distributions.
- Boxplots to identify potential outliers.
- Scatterplots to investigate relationships between income and spending score.
- Grouped statistics to compare spending scores by gender.
- Correlation analysis to investigate linear relationships between numerical features.

### 3. Distribution Analysis and Outlier Detection

I calculated skewness and the interquartile range (IQR) for the numerical features.

Potential outliers were identified using the standard 1.5 × IQR rule.

| Feature | Skewness | Potential Outliers |
|---|---:|---:|
| Age | 0.486 | 0 |
| Annual Income (k$) | 0.322 | 2 |
| Spending Score (1-100) | -0.047 | 0 |

The annual income feature contains two observations flagged by the IQR rule.

Both customers have an annual income of $137,000, but their spending scores differ considerably.

These observations were treated as potential outliers rather than automatically classified as incorrect data.

### 4. Correlation Analysis

I examined the Pearson correlation coefficients between age, annual income, and spending score.

| Feature Pair | Correlation |
|---|---:|
| Age and Annual Income | -0.012 |
| Age and Spending Score | -0.327 |
| Annual Income and Spending Score | 0.010 |

The strongest linear relationship among these pairs was between age and spending score, with a correlation of approximately -0.327.

Annual income and spending score had almost zero linear correlation across the dataset.

This does not mean the features have no relationship of any kind. A correlation coefficient measures linear association and may not reveal more complex patterns.

The scatterplot also showed that customers with similar incomes can have very different spending scores.

### 5. Comparing Spending Scores by Gender

I used grouped statistics to compare the spending scores of male and female customers.

| Statistic | Female | Male |
|---|---:|---:|
| Number of Customers | 112 | 88 |
| Mean Spending Score | 51.53 | 48.51 |
| Median Spending Score | 50.0 | 50.0 |
| Standard Deviation | 24.11 | 27.90 |

Female customers had a slightly higher mean spending score, while both groups had the same median.

Male spending scores had a higher standard deviation, indicating greater variability in this sample.

These are descriptive findings and do not establish that gender causes differences in spending behaviour.

### 6. Feature Selection

For the initial clustering model, I selected two features:

- Annual Income (k$)
- Spending Score (1-100)

CustomerID was excluded because it is an identifier rather than a meaningful behavioural feature.

Age and Gender were also excluded from the initial model. This allowed me to focus on income and spending behaviour and visualise the resulting clusters in two dimensions.

### 7. Feature Scaling

I applied `StandardScaler` before fitting K-Means.

K-Means is a distance-based algorithm. Features with different numerical scales can contribute unequally to distance calculations.

Standardisation transforms the features to have approximately zero mean and unit standard deviation in the fitted dataset.

This allows the initial model to consider income and spending score on comparable scales.

### 8. Selecting the Number of Clusters

I evaluated candidate cluster counts from K = 2 to K = 10 using the Silhouette Score and also examined the Elbow Method.

The Elbow Method uses within-cluster sum of squares (WCSS), also called inertia, to measure the compactness of the clusters.

| Number of Clusters (K) | Silhouette Score |
|---:|---:|
| 2 | 0.321 |
| 3 | 0.467 |
| 4 | 0.494 |
| 5 | 0.555 |
| 6 | 0.540 |
| 7 | 0.528 |
| 8 | 0.455 |
| 9 | 0.457 |
| 10 | 0.443 |

Among the tested values, K = 5 achieved the highest Silhouette Score of 0.555.

The Elbow Method also suggested a noticeable reduction in the marginal improvement in WCSS around this region.

Based on these results, I selected five clusters for the initial model.

The choice was supported by both evaluation methods rather than relying on the Elbow Method alone.

### 9. Final K-Means Model

The final model was trained using the two standardised numerical features.

The implementation used:

- `n_clusters = 5`
- `init = "k-means++"`
- `random_state = 42`
- `n_init = 10`

The model assigned every customer to one of five clusters.

## Results: Customer Segments

The resulting clusters show different combinations of annual income and spending score.

The following profiles are based on the mean age, annual income, and spending score of each cluster.

| Cluster | Customers | Mean Age | Mean Annual Income (k$) | Mean Spending Score |
|---|---:|---:|---:|---:|
| 0 | 81 | 42.72 | 55.30 | 49.52 |
| 1 | 39 | 32.69 | 86.54 | 82.13 |
| 2 | 22 | 25.27 | 25.73 | 79.36 |
| 3 | 35 | 41.11 | 88.20 | 17.11 |
| 4 | 23 | 45.22 | 26.30 | 20.91 |

### Cluster 0: Average-Income, Average-Spending Customers

**Customers:** 81 (40.5%)

- Mean annual income: $55,300
- Mean spending score: 49.52
- Mean age: 42.72

This is the largest cluster. Customers in this group have moderate average income and spending scores.

A business could investigate general loyalty programmes and personalised recommendations for this segment.

### Cluster 1: High-Income, High-Spending Customers

**Customers:** 39 (19.5%)

- Mean annual income: $86,540
- Mean spending score: 82.13
- Mean age: 32.69

This segment combines relatively high annual income with high spending scores.

Possible business applications include premium product recommendations, loyalty programmes, and personalised customer experiences.

### Cluster 2: Low-Income, High-Spending Customers

**Customers:** 22 (11%)

- Mean annual income: $25,730
- Mean spending score: 79.36
- Mean age: 25.27

Customers in this group have lower average incomes but high spending scores.

This segment illustrates why income alone may not be sufficient to understand customer behaviour.

Affordable premium products, targeted rewards, and relevant offers could be investigated for this group.

### Cluster 3: High-Income, Low-Spending Customers

**Customers:** 35 (17.5%)

- Mean annual income: $88,200
- Mean spending score: 17.11
- Mean age: 41.11

This segment has the highest mean annual income among the five clusters but the lowest mean spending score.

A business could investigate whether product preferences, customer experience, or engagement strategies explain this behaviour before deciding on appropriate interventions.

### Cluster 4: Low-Income, Low-Spending Customers

**Customers:** 23 (11.5%)

- Mean annual income: $26,300
- Mean spending score: 20.91
- Mean age: 45.22

Customers in this segment have relatively low annual incomes and spending scores.

Affordable offers and accessible product options could be considered, while recognising that income and spending score alone do not explain individual preferences.

**Note:** Cluster numbers are arbitrary labels assigned by the algorithm. The descriptive names above are based on the observed cluster averages.

The suggested business applications are hypotheses, not proven outcomes. They would require further validation using actual purchasing behaviour and business results.

## Key Learnings

This project helped me understand the importance of analysing data before applying machine learning algorithms.

Key takeaways include:

- Why exploratory data analysis is essential before model training.
- How histograms and boxplots reveal distribution patterns and potential outliers.
- How to calculate and interpret skewness, quartiles, and the interquartile range.
- How grouped statistics can reveal differences between customer groups.
- Why correlation measures linear relationships rather than every possible relationship.
- Why feature scaling matters for distance-based algorithms.
- How to use the Elbow Method and Silhouette Score to compare cluster counts.
- How to interpret cluster profiles instead of treating cluster labels as meaningful by themselves.

## Limitations

- The dataset contains only 200 customers.
- The selected features influence the clusters discovered by K-Means.
- K-Means can be sensitive to outliers, initialisation, and the geometry of the data.
- The Silhouette Score and Elbow Method provide useful evidence but do not guarantee a uniquely correct number of clusters.
- The analysis uses annual income and a mall-assigned spending score rather than detailed purchasing histories.
- The proposed marketing applications have not been tested against real business outcomes.

## Future Improvements

Possible extensions include:

- Compare clustering with and without Age as an additional feature.
- Investigate how outliers affect the resulting clusters.
- Compare alternative preprocessing strategies.
- Experiment with hierarchical clustering and DBSCAN.
- Investigate cluster stability across different initialisations.
- Build an interactive dashboard to explore customer segments.
- Validate proposed marketing strategies against actual customer behaviour.

## How to Run the Project

### Prerequisites

Install Python and the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
