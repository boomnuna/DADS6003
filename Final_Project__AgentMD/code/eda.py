import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# load data 
df = pd.read_csv(r'C:\Users\T00229\Documents\GitHub\DADS6003\Final_Project__AgentMD\dataset\creditcard_train.csv')

#TODO: visualization

# box plot
cols_to_plot = df.columns[1:10]  # Change slice to [10:20] or [20:30] for next batches

plt.figure(figsize=(15, 8))
sns.boxplot(data=df[cols_to_plot])
plt.xticks(rotation=45)
plt.title("Outlier Check for First 10 Columns")
plt.tight_layout()
plt.show()

#TODO: outlier 
# IQR
outlier_counts = {}

# Select only numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    
    # Count how many rows are outside the fences
    num_outliers = df[(df[col] < lower_fence) | (df[col] > upper_fence)].shape[0]
    outlier_counts[col] = num_outliers

# Convert to a DataFrame to look at the results nicely
outlier_df = pd.DataFrame(list(outlier_counts.items()), columns=['Column', 'Outlier Count'])
print(outlier_df.sort_values(by='Outlier Count', ascending=False))

# 1.inpute with meadin, 2.impute with clip, 3.impute with ML
# for outlier >5% or 10% use 1.log transform to move them to center, 2.Robust Scalers

#TODO: feature engineering 
# WOE

#TODO: model
# auto-encoder
