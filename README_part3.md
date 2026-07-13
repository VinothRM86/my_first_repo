Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline
-----------------------------------------------------------
Task 1: Decision Tree baseline:
------------------------------
A baseline Decision Tree Classifier was trained using the default hyperparameters (max_depth=None).
The model achieved:
Training Accuracy: 0.9921
Test Accuracy: 0.8838
Accuracy Gap = 10.83%
A gap of around 11% is a strong indication of overfitting.

The model has memorized details specific to the training data instead of learning patterns that generalize well to new data.
Because the Decision Tree was trained with max_depth=None, the tree continues splitting until it cannot split further.
This creates a very deep and complex tree that fits the training data almost perfectly, learns noise and outliers and
performs worse on unseen data.

Task 2: Controlled Decision Tree
--------------------------------
A second Decision Tree Classifier was trained with controlled complexity using the following hyperparameters:

max_depth = 5
min_samples_split = 20
The purpose of these constraints is to reduce overfitting and improve the model's ability to generalize to unseen data.
Model Configuration

| Parameter         | Value |
| ----------------- | ----- |
| max_depth         | 5     |
| min_samples_split | 20    |
| random_state      | 42    |

Results

| Model                    | Training Accuracy | Test Accuracy | Accuracy Gap |
| ------------------------ | ----------------: | ------------: | -----------: |
| Baseline Decision Tree   |            0.9921 |        0.8838 |       0.1083 |
| Controlled Decision Tree |            0.9305 |        0.9120 |       0.0185 |

Interpretation:

The parameter max_depth limits how deep the tree is allowed to grow. 
Restricting the depth prevents the model from creating highly specific decision rules, thereby reducing model variance and the likelihood of overfitting. 
Although this may introduce a small amount of bias, it generally improves the model's ability to generalize.
The parameter min_samples_split specifies the minimum number of samples required to split an internal node. 
By requiring at least 20 samples before a split can occur, the model avoids creating branches based on very small subsets of data, 
which are more likely to represent noise than meaningful patterns.

Compared with the baseline Decision Tree, the controlled tree should exhibit a smaller gap between training and test accuracy, 
indicating reduced overfitting and better generalization. While the training accuracy may decrease slightly, 
the test accuracy often remains similar or improves because the model focuses on broader, more robust patterns rather than memorizing the training data.

Conclusion
The controlled Decision Tree provides a better balance between model complexity and predictive performance. 
By limiting tree growth and preventing unnecessary splits, it produces a more stable model that is less sensitive to variations in the training data.

Task 3: Gini vs Entropy comparison
----------------------------------
Objective
Two Decision Tree models were trained using different splitting criteria:
Gini Impurity
Entropy (Information Gain)
Both models were configured with max_depth = 5 and random_state = 42 to ensure a fair comparison.

Results
  Criterion  Test Accuracy
0      Gini         0.9085
1   Entropy         0.9085

Interpretation
A node with Gini = 0 is a pure node, meaning that all samples within the node belong to a single class. Since there is no class uncertainty, no further splitting is required.

Gini Impurity and Entropy are both measures of node impurity used to determine the best split in a Decision Tree. 
Gini Impurity is computationally simpler because it does not require logarithmic calculations, whereas Entropy uses information gain and may be slightly slower to compute. 
In practice, both methods often produce very similar tree structures and predictive performance.
Conclusion
For this dataset, both splitting criteria produced comparable classification accuracy, indicating that either impurity measure is suitable. 
Since the performance difference is minimal, the choice between Gini and Entropy can be based on computational efficiency or project requirements rather than predictive performance alone.

Task 4: Random Forest
---------------------
A Random Forest Classifier was trained using the following hyperparameters:

n_estimators = 100
max_depth = 10
random_state = 42

The model was evaluated using Training Accuracy, Test Accuracy, and ROC-AUC.
Results

| Metric            |      Value |
| ----------------- | ---------: |
| Training Accuracy |   95.95%   |
| Test Accuracy     |   91.20%   |
| ROC-AUC           |   0.9312   |

The Random Forest achieved a higher test accuracy than the Decision Tree models while maintaining a high ROC-AUC score, indicating strong predictive performance and good generalization.

### Top 5 Important Features

| Rank | Feature    | Importance |
| ---- | ---------- | ---------: |
| 1    | age        | 0.5048     |
| 2    | smoker_yes | 0.2959	 |
| 3    | bmi        | 0.1181	 |
| 4    | children   | 0.0436	 |
| 5    | sex_male   | 0.0143	 |

The results indicate that age is the most influential predictor of the target class, followed by "smoking status" and "BMI". 
These variables contribute the most to the model's decision-making process.

Feature Importance
Random Forest computes feature importance by measuring the average reduction in Gini impurity whenever a feature is used to split a node. 
Each split decreases the impurity of the resulting child nodes, and these reductions are accumulated across all trees in the forest. 
Features that consistently produce larger impurity reductions receive higher importance scores.

Task 4a: Gradient Boosting
--------------------------
A Gradient Boosting Classifier was trained using the following hyperparameters:

n_estimators = 100
learning_rate = 0.1
max_depth = 3
random_state = 42

The model was evaluated using Training Accuracy, Test Accuracy, and ROC-AUC.

Results
| Metric            |      Value |
| ----------------- | ---------: |
| Training Accuracy | 94.81% 	|
| Test Accuracy     | 91.90%	|
| ROC-AUC           |	0.9422  |

Interpretation
Interpretation of Results
Metric			Value	Interpretation
Training Accuracy	95.95%	Very good fit on the training data.
Test Accuracy		91.20%	Better than both Decision Trees, indicating strong generalization.
ROC-AUC			0.9312	Excellent ability to distinguish between the two classes.

Interpretation
Training Accuracy = 94.81%
The model fits the training data well without memorizing it.
Test Accuracy = 91.90%
This is the highest test accuracy you've achieved so far.
ROC-AUC = 0.9422
A ROC-AUC of 0.9422 indicates excellent discrimination between the two classes. 
In practical terms, the model has about a 94.22% chance of ranking a randomly chosen positive instance higher than a randomly chosen negative instance.
Compared with the Random Forest:

Metric	Random Forest	Gradient Boosting
Training Accuracy	95.95%	94.81%
Test Accuracy		91.20%	91.90% 
ROC-AUC			0.9312	0.9422 

Although Gradient Boosting has a slightly lower training accuracy, it achieves better test accuracy and ROC-AUC, suggesting it generalizes better to unseen data.

Task 4b: Feature ablation study
-------------------------------
Evaluate whether the five least important features identified by the Random Forest model contribute meaningfully to predictive performance.

Steps:
- Identify the five least important features.
- Remove them from both training and test datasets.
- Train another Random Forest using identical hyperparameters.
- Compare the ROC-AUC of the full model and the reduced model.

Results:
Features Removed:
Feature			Importance
children		0.043603
sex_male		0.014345
region_southwest	0.008327
region_southeast	0.008176
region_northwest	0.006735

These were the five least important features according to the Random Forest model.
Model Comparison
Model			ROC-AUC
Full Random Forest	0.931151
Reduced Random Forest	0.927183

Difference = 0.0040
A decrease of 0.004 in ROC-AUC is very small
This means, removed features do contain some predictive information. However, their contribution is minor.
The majority of the model's predictive power comes from:
age
smoker_yes
bmi

Conclusion:
The removed features contributed a small amount of predictive information. Removing them slightly reduced model performance, but the degradation was minimal

Task 5: Cross-validated comparison

Objective
Four classification models were evaluated using 5-fold Stratified Cross-Validation with ROC-AUC as the evaluation metric.
The models compared were:
Logistic Regression
Controlled Decision Tree
Random Forest
Gradient Boosting

Results:
| Rank | Model                    | Mean ROC-AUC | Std ROC-AUC |
| ---- | ------------------------ | -----------: | ----------: |
| 1    | Random Forest	          |   0.9456	 |  0.0062     |
| 2    | Logistic Regression      |   0.9435     |  0.0046     |
| 3    | Gradient Boosting        |   0.9356     |  0.0056     |
| 4    | Controlled Decision Tree |   0.9278     |  0.0113     |

Interpretation
Random Forest
Highest average ROC-AUC (0.9456)
Small standard deviation (0.0062)

This indicates:
Excellent predictive performance
Very stable across different folds
Strong generalization ability

Logistic Regression
Mean AUC = 0.9435
This is close to Random Forest. It means dataset has a fairly linear decision boundary.
Gradient Boosting

Single Test ROC-AUC:0.9422
Cross Validation Mean:0.9356
This shows why Cross Validation is useful.
Although Gradient Boosting looked best on one train-test split, across five different splits it performs slightly worse than Random Forest.
Controlled Decision Tree:
Mean AUC: 0.9278
Highest Std: 0.0113

Meaning:
More unstable
Performance varies depending on data split
Still much better than the unrestricted tree
Random Forest is actually more reliable, because Cross Validation tests the model on 5 different train/test combinations, 
reducing the chance that one lucky or unlucky split influences the evaluation.

Task 6: Hyperparameter tuning with GridSearchCV
-----------------------------------------------
Hyperparameter tuning was performed on the Random Forest classifier using GridSearchCV. A machine learning pipeline was constructed consisting of:

SimpleImputer(strategy='median')
StandardScaler()
RandomForestClassifier(random_state=42)

The model was evaluated using 5-fold Stratified Cross-Validation with ROC-AUC as the scoring metric.
Result:
Best parameters:
n_estimators = 50
max_depth = 10
min_samples_leaf = 1

The algorithm found that 50 trees were sufficient. Adding more trees (100 or 200) did not improve the cross-validation ROC-AUC 
enough to justify their additional computational cost.

Best Cross-Validation ROC-AUC: 0.948712786463911
A ROC-AUC of 0.9487 indicates that the tuned Random Forest has approximately a 94.87% probability 
of ranking a randomly selected positive instance higher than a randomly selected negative instance.

18 parameter combinations, Each was evaluated with 5-fold Cross Validation = 90 trainings
Comparison with Previous Random Forest
Model	Mean CV ROC-AUC
Random Forest (before tuning)		0.9456
Random Forest (after Grid Search)	0.9487

improvement: 0.0031
Although the improvement is modest, it demonstrates that hyperparameter tuning can produce a better-performing model.

Task 7: Manual learning curve
-----------------------------
Results:
| Training Fraction | Training AUC | Test AUC |
| ----------------: | -----------: | -------: |
|               20% |       1.0000 |   0.9282 |
|               40% |       0.9999 |   0.9351 |
|               60% |       0.9996 |   0.9266 |
|               80% |       0.9990 |   0.9407 |
|              100% |       0.9983 |   0.9309 |

Training AUC decreases. As more training data is added, the Random Forest cannot memorize every training example as easily, 
so the training AUC decreases slightly. However, it remains extremely high (above 0.998), indicating that the model fits the training data very well.

Test AUC not consistently increases could due to smaller rows around 1340.
Final Test AUC = 0.9309 is lower than the best value obtained earlier 0.9407
The model is not obviously limited by the amount of training data. 
Instead, the model appears to have reached a performance plateau, meaning that simply collecting more data may not consistently improve performance. 
Further improvements are more likely to come from better feature engineering, additional predictive variables, or trying different model architectures.

Task 8: Serialize the best model
--------------------------------

Results:
Manual Learning Curve
============================================================
   Training Fraction  Training AUC  Test AUC
0                0.2        1.0000    0.9282
1                0.4        0.9999    0.9351
2                0.6        0.9996    0.9266
3                0.8        0.9990    0.9407
4                1.0        0.9983    0.9309

Model saved successfully as best_model.pkl
joblib.dump(best_pipeline, "best_model.pkl")
Load the model
loaded_model = joblib.load("best_model.pkl")

Prediction on New Data
Two manually created test samples with the same feature structure used during model training were provided to the loaded model.
new_data = pd.DataFrame({
    "age": [30, 55],
    "bmi": [24.5, 36.8],
    "children": [1, 3],
    "sex_male": [1, 0],
    "smoker_yes": [0, 1],
    "region_northwest": [1, 0],
    "region_southeast": [0, 1],
    "region_southwest": [0, 0]
The model successfully predicted the following class labels:

Test Sample	Prediction
Sample 1	0
Sample 2	1

Where:
0 represents insurance charges less than or equal to the median.
1 represents insurance charges greater than the median.

The successful predictions confirm that the serialized model can be loaded and used without retraining, 
demonstrating that the complete machine learning pipeline has been preserved correctly.

Task 9: Summary comparison table
--------------------------------
Model Performance Comparison
Model				5-Fold CV Mean ROC-AUC	5-Fold CV Std ROC-AUC	Test ROC-AUC
Logistic Regression		0.9435			0.0046			0.9203
Controlled Decision Tree	0.9278			0.0113			N/A 
Random Forest			0.9456			0.0062			0.9312
Gradient Boosting		0.9356			0.0056			0.9422
Tuned Random Forest(GridSearchCV)0.9487			5-fold CV	(Insert tuned model test AUC if computed)

Tuned Random Forest is recommended for deployment









