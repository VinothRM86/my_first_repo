Project Name: Medical Insurance Cost Prediction.
Dataset downloaded from Kaggle and dataset is available Github.
Part 1 is to prepare the raw dataset for machine learning by performing data cleaning, EDA analysis, Statistical and Visualization analysis.
Dataset column/description
Column	    Description
age	        age of the insured person
sex	        Gender (Male/Female)
bmi	        Body Mass Index
children	Number of dependent children
smoker	    Smoking status
region	    Residential region
charges	    Annual medical insurance cost (Target Variable)
Task 1: Load dataset
--------------------
Dataset has been loaded into Pandas dataframe using pd.read_csv
Display 1st five rows using head()
checked and displayed data types using dtypes
verified the no.of rows and columns using shape 
Dataset Shape: (2794, 7)

Task 2: Null value analysis
---------------------------
null values have been identified using df.isnull().sum()
null percentage have calculated using (df.isnull().sum() / df.shape[0]) * 100
No columns have more than 20% null values.
So, numeric columns containing less than 20% null values are replaced using the median.
Why Median Instead of Mean?
The mean is sensitive to extreme values, but the median is resistant to outliers and therefore median provides a better central tendency.
For the above reason, we chosen median imputation.

Task 3: Duplicate detection and removal
---------------------------------------
Duplicate records were identified using df.duplicated().sum()
Duplicate rows were removed using df.drop_duplicates()
The number of duplicate removed rows : 1374
After removing duplicates, null percentages were recalculated and no unexpected changes.

Task 4:Data type correction
---------------------------
Data types were reviewed after loading the dataset.
The following preprocessing was performed:
Numeric columns stored as object were converted using pd.to_numeric().
Repetitive text columns like sex, smoker and region were converted into the category data type.
Memory usage was compared before and after conversion using df.memory_usage(deep=True).sum()
Memory usage before conversion: 288,541 bytes
Memory usage after conversion : 61,891  bytes
Converting categorical columns reduced memory consumption and improved storage efficiency.

Task 5: Descriptive statistics and skewness
-------------------------------------------
Descriptive statistics were generated using df.describe()
For every numeric column, skewness was calculated using df.skew()
The numeric column with the highest absolute skewness was identified and the column is Charges.
Positive skewness indicates that the distribution has a long right tail caused by large values.
Negative skewness indicates that the distribution has a long left tail caused by small values.
Because the mean shifts toward extreme values, due to this median imputation was preferred for skewed variables.

Task 6: Outlier Detection Using IQR
-----------------------------------
Outliers were identified for selected numeric variables using the Interquartile Range (IQR) method.
The following values were calculated:
First Quartile (Q1) 		- 26.31
Third Quartile (Q3)		- 34.68
Interquartile Range (IQR)	- 8.37
Lower Bound = Q1 − 1.5 × IQR	- 13.75
Upper Bound = Q3 + 1.5 × IQR	- 47.24

Rows outside these limits were counted as outliers - 11
Outlier Handling:
Outliers were not removed during Part 1 because they may represent valid medical cases and it's retained.
If required, outlier treatment such as capping or transformation can be applied during feature engineering in Part 2.

Task 7: Visualizations
---------------------- Impu
Line plot: A line plot of the insurance was generated using dataset row index and it helps to identify overall trends.
Bar Chart: A bar chart compared the average insurance charges across different categories.Eg: Smoker and Non-smokers
The chart highlighted differences between categorical groups.
Histogram: A histogram of the charges column shows a positively skewed distribution.
This distribution showed clear skewness and it confirms in the skewness calculation.
Scatter Plot: A scatter plot was generated between BMI and insurance charges indicates positive relationship.
As BMI increases, insurance charges generally increases. However, the relationship is not perfectly linear because several other factors, 
such as smoking status and age, also influence insurance costs.
Box Plot: The box plot compares insurance charges for smokers and non-smokers.
Smokers exhibit a much higher median insurance charge and a larger spread than non-smokers. Several high-cost outliers are also visible among smokers, 
suggesting greater variability in medical expenses for this group.
The visualization clearly displayed:
Median differences
Presence of outliers

Task 8: Correlation heatmap
---------------------------
Pearson correlation coefficients were calculated for all numeric variables. The absolute correlation was observed between age and charges: value is 0.32
The correlation matrix was visualized using a heatmap.
The pair of variables with the strongest absolute correlation was identified.
Interpretation
A strong correlation does not necessarily imply causation.
For example, insurance charges  highly correlated with smoking status, but other factors such as age,
BMI, lifestyle, or underlying medical conditions may also contribute to higher medical costs.

Task 9a: Imputation stratergy comparison
---------------------------------------
The mean and median were compared for the two most skewed numeric columns before missing value imputation.
For positively skewed distributions, extreme high values increase the mean, making it less representative of the typical observation. 
Therefore, the median was selected because it is robust to outliers.
If a column had been negatively skewed, the mean would have been influenced by small values, making the median the preferred measure in that case as well.
Consequently, missing values in the selected columns were imputed using the median.
Task 9b: Spearman Rank Correlation
----------------------------------
Spearman correlation measures monotonic relationships by comparing the ranks of values rather than the values themselves.
This analysis compares Spearman correlation with Pearson correlation to identify non-linear but consistent relationships.
Task 9c: Grouped aggregation
----------------------------
Insurance charges were grouped by smoking status.The smoker group had the highest average insurance charges as well as the highest standard deviation.
The higher within-group standard deviation indicates that smoking status alone is insufficient to accurately predict insurance charges 
because considerable variation still exists within the smoker group.
The ratio between the highest and lowest group means was 3.73, indicating that smoking status provides a strong predictive signal 
and should be retained as an important feature for machine learning in Part 2.

Task 10: Save the clean dataset using df.to_csv for further processing
