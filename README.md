# Predicting-Heart-Attacks
Predicting Risk of Cardiac Arrhythmia using Machine Learning Algorithms

# Introduction
---
At this moment, cardiovascular diseases represent the
first mortality cause in humans. So, it is becoming more and
more critical for machines to predict such disasters before
they happen.
This paper aims at a better understanding and application
of machine learning in medical domain. In this paper, we
modify six classical models for multi-class problems such
as: Logistic Regression, Naive Bayes and SVM, and then
implement them to predict cardiac arrhythmia based on pa-
tients’ medical records.

# Dataset
---
We use the UCI Arrhythmia Data Set for both training and testing. 
Weare provided with 452 clinical records of patients. Each
record contains 279 attributes, such as age, sex, weight and
information collected from ECG signals. The diagnosis of
cardiac arrhythmia is divided into 16 classes. 
Class 1 refers to a normal case. 
Class 2 to 15 represent different kinds
of cardiac arrhythmia, such as Ischemic Changes, Old Anterior 
Myocardial Infarction, Supra-ventricular Premature
Contraction, Right Bundle Branch Block and etc. 
Class 16 refers to the rest.

The Dataset can be found at the [UCI Data Repository] (http://archive.ics.uci.edu/ml/datasets/Arrhythmia)

# Algorithms Used
---
1. Logistic Rgeression
2. Multinomial Naive Bayes
3. Decision Trees
4. Random Forest
5. K-Nearest Neighbours
6. Support Vector Machines

# Feature Selection
---
Feature selection was done according to the k highest scores.

# Results
---
|     Model     | Accuracy before | No. of Features selected  | Accuracy after |
| --- | --- | --- | --- |
|     Logistic Regression     | 63.97 % | 75  |  65.44 %|
|     Multinomial Naive Bayes     | 58.82 % | 76  | 62.5 %  |
|     Decision Trees     | 59.55 %  |  40 | 63.97 %  |
|     Random Forest     | 67.64 %  |  126 | 71.32 %  |
|     K-Nearest Neighbours     | 58.08 % | 50  | 66.176 % |
|     Support Vector Machine     | 58.82 % | 20  | 66.176 % |

# References
---
[1] Guvenir, H. Altay, et al. ”A supervised machine learning
algorithm for arrhythmia analysis.” Computers in Cardi-
ology 1997. IEEE, 1997.

[2] Mishra, Binod Kumar, Prashant Lakkadwala, and
Naveen Kumar Shrivastava. ”Novel Approach to Predict
Cardiovascular Disease Using Incremental SVM.” Com-
munication Systems and Network Technologies (CSNT),
2013 International Conference on. IEEE, 2013.

[3] Chen, Luyang, Q. Cao, S. Li and Xiao Ju. “Predicting
Heart Attacks.” (2014).

# Team:
---
Adwiteeya Chaudhry<br/>
Ayush Bharadwaj<br/>
Paras Chaudhry
