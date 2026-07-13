Part 2: Supervised Machine Learning Model
Task 1: Load dataset
--------------------
Load dataset using pd.read_csv

Target Variable Definition
Two prediction tasks were created from the cleaned dataset.
Regression Target (y_reg)
The regression target variable is "charges", representing the medical insurance cost for each individual.
y_reg = df["charges"]
Classification Target (y_clf)
A binary classification target was created by comparing each insurance charge with the median charge value.
y_clf = (y_reg > y_reg.median()).astype(int)
Class labels:

0 - Insurance charge is less than or equal to the median (Low Charges)
1 - Insurance charge is greater than the median (High Charges)
This approach creates a balanced binary classification problem while preserving the original regression task.

Task 2:Encode categorical columns
Categorical Feature Encoding
The feature matrix contained three categorical variables:
sex
smoker
region
These variables doesn't have a natural order. Therefore, One-Hot Encoding was applied using: pd.get_dummies(drop_first=True)
The first dummy variable was dropped to avoid the Dummy Variable Trap(perfect multicollinearity).
Label encoding wasn't used because assigning integers such as:
- Male = 0, Female = 1
- Northeast = 0, Northwest = 1, Southeast = 2, Southwest = 3
would incorrectly imply an ordinal relationship between categories. 
One-Hot Encoding represents each category independently, preventing the model from interpreting one category as being "greater" or "less" than another.
Dataset doesn't contaion ordinal categorical values such as low,medium, high. So there is no need of Lable encoding.

Task 3: Leak-free train-test split and scaling
----------------------------------------------
Train-Test Split and Feature Scaling
The dataset was divided into training and testing sets using an 80:20 split with:
train_test_split(test_size=0.2, random_state=42)
Feature scaling was performed using 'StandardScaler'.
The scaler was "fitted only on the training data": scaler.fit(X_train)
The fitted scaler was then used to transform both the training and testing datasets:
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
This approach prevents data leakage. If the scaler were fitted on the entire dataset before splitting, 
the mean and standard deviation would include information from the test set. 
This would allow the model to indirectly learn characteristics of the test data during training.

Task 4: Regression model — Linear Regression
--------------------------------------------
The model coefficients indicate how each standardized feature influences the predicted insurance charges while holding all other features constant.
A positive coefficient means that increasing the standardized feature by one unit increases the predicted insurance charge by about the coefficient value.
A negative coefficient means that increasing the standardized feature by one unit decreases the predicted insurance charge by about the coefficient value.

The three largest absolute coefficients represent the features with the strongest influence on the model's predictions.

Ridge Regression

Ridge Regression applies L2 regularization, which adds a penalty to large coefficient values.
Unlike Linear Regression, Ridge Regression discourages overly large coefficients, making the model more stable when features are correlated.

The alpha parameter controls the strength of the regularization:
A small alpha results in behavior similar to Linear Regression.
A Large alpha shrinks coefficients more aggressively, which can reduce overfitting but may also increase bias.

In this project, alpha = 1.0 is used. The performance of Ridge Regression was compared with Linear Regression using MSE and R² Score.
Linear Regression Performance
-----------------------------
MSE : 42532056.10
R²  : 0.7265

Ridge Regression Performance
----------------------------
MSE : 42530268.49
R²  : 0.7265

Task 5: Classification model — Logistic Regression
--------------------------------------------------
The class distribution was examined before model training.
Since the binary target was created using the median of the insurance charges, both classes contained approximately equal numbers of observations.
Therefore, no imbalance handling techniques such as SMOTE was required.
Precision Formula = TP/(TP+FP)
Recall Formula = TP/(TP+FN)

Recall is considered slightly more important than Precision for this project because failing to identify an individual 
with genuinely high insurance charges (False Negative) may lead to underestimating expected medical costs. 
Although false positives are undesirable, identifying most high-cost individuals is generally more valuable.

AUC Means:
The Area Under the ROC Curve (AUC) measures the model's ability to distinguish between the two classes across all possible decision thresholds.
An AUC value close to 1.0 indicates excellent discrimination, while a value near 0.5 indicates performance similar to random guessing.
The obtained AUC demonstrates how effectively the Logistic Regression model separates individuals with low and high insurance charges.
AUC score is 0.9203

Decision-threshold sensitivity:
The decision threshold was varied from 0.30 to 0.70 to evaluate its effect on Precision, Recall, and F1-score.
	Threshold	Precision	Recall	F1 Score
0	0.3	0.820988	0.880795	0.849840
1	0.4	0.880000	0.874172	0.877076
2	0.5	0.935714	0.867550	0.900344
3	0.6	0.947368	0.834437	0.887324
4	0.7	0.983333	0.781457	0.870849

The highest F1-score (0.9003) was achieved at the default threshold of 0.50. 
This indicates that the default threshold provides the best balance between precision and recall for this dataset.
As the decision threshold increased, precision improved because the model became more conservative when predicting the positive class. 
However, recall decreased because more actual positive cases were missed. 
Lowering the threshold increased recall but also increased the number of false positives, reducing precision.

For this medical insurance classification task, recall is considered slightly more important 
because failing to identify an individual with genuinely high insurance charges (a false negative) may underestimate future insurance costs. 
If maximizing recall were the primary objective, the threshold could be lowered below 0.50. 
However, this would increase the number of false positives and reduce precision. 
Since the highest F1-score occurs at 0.50, this threshold was selected as it provides the best overall balance between precision and recall.

Task 6: Regularization experiment on Logistic Regression:
---------------------------------------------------------
Logistic Regression model was trained using C = 0.01 and compared with the baseline model using C = 1.0.

	Metric		C=0.01		C=1.0
0	Precision	0.928058	0.935714
1	Recall		0.854305	0.867550
2	AUC		0.918438	0.920281

The parameter C controls the inverse of the regularization strength in Logistic Regression. 
A smaller value of C applies stronger L2 regularization, shrinking the model coefficients toward zero and reducing model complexity. 
This can help prevent overfitting but may also lead to underfitting if the regularization is too strong.

In this project, reducing C from 1.0 to 0.01 resulted in a slight decrease in Precision, Recall, and AUC. 
This indicates that stronger regularization reduced the model's ability to capture useful relationships in the data. 
Since the baseline model consistently achieved better performance across all evaluation metrics, C = 1.0 was selected as the preferred Logistic Regression model for this dataset.

Task 7: Bootstrap confidence interval for AUC difference
--------------------------------------------------------
Bootstrap resampling with 500 iterations was used to estimate how consistently the Logistic Regression model with C = 1.0 outperformed the model with C = 0.01.

For each bootstrap sample, the difference in AUC (AUC of C = 1.0 minus AUC of C = 0.01) was calculated.

The results were:

Mean AUC Difference: 0.001961
95% Confidence Interval:[-0.001178, 0.005307]

The mean AUC difference indicates that the C = 1.0 model achieved an average AUC approximately 0.001961 higher than the C = 0.01 model. 
However, the 95% confidence interval includes zero, which means the observed difference is not statistically significant at the 95% confidence level.

Therefore, although the baseline model (C = 1.0) achieved slightly higher Precision, Recall, and AUC on the test set, 
the bootstrap analysis suggests that this advantage may be due to sampling variability rather than a consistent improvement. 
Based on this analysis, there is insufficient statistical evidence to conclude that the C = 1.0 model reliably outperforms the C = 0.01 model across different samples of data.






