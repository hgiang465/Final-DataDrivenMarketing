# Final Data-Driven Marketing Analytics
## Project Information

**Subject**: Data-Driven in Marketing  
**Instructor**: Dr. Nguyen Tuan Long

**Team Members**:

| No. | Name | Student ID | Class |
|-----|------|------------|-------|
| 1 | Ninh Duy Tuan | 11230603 | DSEB 65B |
| 2 | Nguyen Thi Huong Giang | 11230521 | DSEB 65B |
| 3 | Nguyen Thanh Mo | 11230571 | DSEB 65B |
| 4 | Le Lan Huong | 11230561 | DSEB 65B |
| 5 | Nguyen Khanh Huyen | 11230545 | DSEB 65B |


## Project Overview

This project implements an end-to-end data-driven marketing solution that transforms raw customer data into actionable business insights. The analysis includes exploratory data analysis, customer segmentation, churn prediction, customer lifetime value forecasting, and A/B testing evaluation.

## Key Objectives

- **Customer Understanding**: Segment customers into meaningful groups for targeted marketing
- **Churn Prediction**: Identify at-risk customers to enable retention strategies
- **Revenue Optimization**: Predict customer lifetime value for resource allocation
- **Experimentation**: Conduct and analyze A/B tests to improve marketing effectiveness
- **Data Quality**: Ensure data integrity through comprehensive cleaning and validation

## Data Dictionary
This dataset provides a comprehensive overview of e-commerce operations throughout the fiscal year 2019 (January 1st to December 31st). It captures 52,924 individual transactions involving 1,468 unique customers.

| Column Name | Description | Data Type |
| :--- | :--- | :--- |
| **CustomerID** | Unique identifier for each customer | `int64` |
| **Transaction_ID** | Unique identifier for each transaction | `int64` |
| **Transaction_Date** | Date when the transaction occurred (YYYY-MM-DD) | `string` |
| **Product_SKU** | Unique Stock Keeping Unit (Product ID) | `string` |
| **Product_Description** | Full name/detailed description of the product | `string` |
| **Product_Category** | High-level category of the product | `string` |
| **Quantity** | Number of units ordered in the transaction | `int64` |
| **Avg_Price** | Average price per single unit | `float64` |
| **Delivery_Charges** | Charges for delivering the order | `float64` |
| **Coupon_Status** | Indicates if a discount coupon was applied | `string` |
| **Gender** | Gender of the customer | `string` |
| **Location** | Geographical location of the customer | `string` |
| **Tenure_Months** | Total months the customer has been with the platform | `int64` |
| **Month** | Month when the promotional coupon is applicable | `string` |
| **Coupon_Code** | Specific promotional code for a category | `string` |
| **Discount_pct** | Percentage of discount offered | `int64` |
| **Date** | Reference date for marketing spend | `string` |
| **Offline_Spend** | Daily marketing spend on offline channels (TV, Radio, etc.) | `int64` |
| **Online_Spend** | Daily marketing spend on online channels (Google, FB, etc.) | `float64` |
| **GST** | Goods and Services Tax percentage for the category | `float64` |


## Project Structure

```
Final-DataDrivenMarketing/
├── data_cleaning.ipynb              # Data preprocessing and validation
├── feature_engineering.ipynb         # Feature creation and transformation
├── EDA_business_insights.ipynb       # Exploratory data analysis with visualizations
├── customer_table.ipynb              # Customer data aggregation and enrichment
├── customer_segmentation.ipynb       # RFM analysis and clustering
├── churn_prediction.ipynb            # Predictive modeling for churn risk
├── CLV_Prediction.ipynb              # Customer lifetime value forecasting
├── A_B_testing.ipynb                 # Statistical analysis of A/B tests
├── Dataset/                          # Raw and processed data files
├── Output/                           # Generated reports and visualizations
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies Include
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- scikit-learn: Machine learning models
- matplotlib & seaborn: Data visualization
- scipy: Statistical analysis
- statsmodels: Statistical modeling

## Quick Start

1. **Explore the data**:
   ```bash
   jupyter notebook data_cleaning.ipynb
   ```

2. **Review business insights**:
   ```bash
   jupyter notebook EDA_business_insights.ipynb
   ```

3. **Run predictive models**:
   ```bash
   jupyter notebook churn_prediction.ipynb
   jupyter notebook CLV_Prediction.ipynb
   ```

4. **View segmentation results**:
   ```bash
   jupyter notebook customer_segmentation.ipynb
   ```

## Key Findings & Insights

This project delivers actionable insights through:
- **Customer segments** for targeted marketing campaigns
- **Churn risk scores** for retention program prioritization
- **Revenue predictions** for financial forecasting
- **A/B test results** for campaign optimization
- **Business recommendations** based on data analysis

## Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning
- **Matplotlib/Seaborn**: Visualization
- **Jupyter**: Interactive analysis notebooks
- **SciPy/StatsModels**: Statistical analysis

## Data Directories

- **Dataset/**: Contains all input data files required for analysis
- **Output/**: Stores generated reports, predictions, and visualizations

