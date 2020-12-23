# =============================================================================
# Team - 38
# Adwiteeya Chaudhry (Roll No. 2018126)
# Ayush Bharadwaj (Roll No. 2018134)
# Paras Chaudhary (Roll No. 2018167)
# Indraprashtha Institute of Information Technology, Delhi
# Machine Learning Project
# Predicting Risk of Cardiac Arryhthmia using Machine Learning Techniques
# Implementation of ML Models on the Dataset along with Feature Selection
# =============================================================================

import time

# starting time
start = time.time()

# =============================================================================
#                           IMPORTING LIBRARIES                               #
# =============================================================================

import pickle
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (12,8);

# =============================================================================
#                            DATA PREPROCESSING                               #
# =============================================================================

df = pd.read_csv("Data/Arrhythmia_Dataset.csv")
# df.head()

# Dividing the classes between Normal ECG class of Arrhythmia (1) 
# and all the other classes of Arrhythmia (0)   

# df.loc[df['CLASS'] != 1, 'CLASS'] = 0

df = df.drop(df.index[4])
df[['Heart Rate']]

# Removing the 5 features (out of 279) that contain missing values

df = df.drop(['T','P','QRST','J'], axis = 1)
# df = df.astype('float64')

# Class counts
df = df.astype('int')
df['CLASS'].value_counts()

##### IGNORE #####
df.iloc[:,2]
df.shape[1]
##### IGNORE #####

# Discretizing the Data (to avoid problems during Naive Bayes)

features = df.shape[1] - 1
for  i in range(features):
    df.iloc[:,i] = pd.cut(df.iloc[:,i], bins=10, labels=False, right=False)
df.head()

# Incrementing all the classes of the features 
# This ensures that the bins range from 1 to 10 (So, 0 doesn't interfere)

df.iloc[:,:-1] += 1

# Convert the final Class Column into integers (Uniformity of Datatype in the dataframe)

df.iloc[:,-1] = df.iloc[:,-1].astype('int64')

# Randomizing the Data. Just in case ...

df = df.sample(frac = 1).reset_index(drop=True)    ## Reset index ensures we don't get another index columns for the new indexes
# df

# =============================================================================
#                      MODELS WITHOUT FEATURE SELECTION                       #
# =============================================================================

# Initialization of input vectors X and y

# Initializing X and y 
X = df[df.columns[:-1]].values            # Number of samples x Number of features
y = df[df.columns[-1]].values             # 1D Array containing all class values
print()
print("Size of Patient records :", X.shape)
print("Number of Class Records :", y.shape)

print()
print("                         WITHOUT FEATURE SELECTION")
print()
# Train-Test Split (70 : 30)
# Stratified Split

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5, test_size = 0.3,stratify = y)

############################### NAIVE BAYES ###################################

print("NAIVE BAYES")

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

file = "Weights/Naive_Bayes_Model.sav"
pickle.dump(clf, open(file, 'wb'))
        
print()
print("SkLearn's Multinomial Naive Bayes Classification Accuracy :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()

############################## RANDOM FORESTS #################################

print("RANDOM FORESTS")

from sklearn.ensemble import RandomForestClassifier 
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test) 

file = "Weights/Random_Forest_Model.sav"
pickle.dump(clf, open(file, 'wb'))

print()
print("SkLearn's Random Forest Classification Accuracy :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()

############################## LOGISTIC REGRESSION ############################

print("LOGISTIC REGRESSION")

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver = 'liblinear')
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test) 

file = "Weights/Logistic_Regression_Model.sav"
pickle.dump(clf, open(file, 'wb'))

print()
print("SkLearn's Logistic Regression Classification Accuracy :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()

############################## K-NEAREST NEIGHBORS ###########################

print("K-NEAREST NEIGHBORS")

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=4)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test) 

file = "Weights/K_Nearest_Neighbors_Model.sav"
pickle.dump(clf, open(file, 'wb'))

print()
print("SkLearn's K-Nearest Neighbors Classification Accuracy :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()

############################## DT ###########################

print("DT")

from sklearn.datasets import load_iris 
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
file = "Weights/DT_Model.sav"
pickle.dump(clf, open(file, 'wb'))

print()
print("SkLearn's DT Classification Accuracy :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()

############################## Multinomial Naive Bayes ###########################

print("Multinomial Naive Bayes")

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
file = "Weights/Multinomial_Naive_Bayes_Model.sav"
pickle.dump(clf, open(file, 'wb'))

print()
print("SkLearn's Multinomial Naive Bayes Classification Accuracy :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()

############################## SVC ###########################

print("SVC")

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
file = "Weights/SVC_Model.sav"
pickle.dump(clf, open(file, 'wb'))

print()
print("SkLearn's SVC Classification Accuracy :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()



# =============================================================================
#                        MODELS WITH FEATURE SELECTION                        #
# =============================================================================

print()
print("                            WITH FEATURE SELECTION")
print()

############################### NAIVE BAYES ###################################

import sklearn
X_new = sklearn.preprocessing.MinMaxScaler().fit_transform(X)

from sklearn.feature_selection import SelectKBest, chi2
n_features = 76
X_new = SelectKBest(chi2, k = n_features).fit_transform(X_new, y)

print("NAIVE BAYES")
print("Number of Features Selected :", n_features)

# Train-Test Split (70 : 30)
# Stratified Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state = 0, test_size = 0.3,stratify = y)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

file = "Weights/NB_Feature_Extraction_Model.sav"
pickle.dump(clf, open(file, 'wb'))

print()
print("Multinomial Naive Bayes Classification Accuracy after Feature Extraction :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()

############################## RANDOM FORESTS #################################

import sklearn
X_new = sklearn.preprocessing.MinMaxScaler().fit_transform(X)

from sklearn.feature_selection import SelectKBest, chi2
n_features = 126
X_new = SelectKBest(chi2, k = n_features).fit_transform(X_new, y)

print("RANDOM FORESTS")
print("Number of Features Selected :", n_features)

# Train-Test Split (70 : 30)
# Stratified Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state = 0, test_size = 0.3,stratify = y)

from sklearn.ensemble import RandomForestClassifier 
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test) 

file = "Weights/RF_Feature_Extraction_Model.sav"
pickle.dump(clf, open(file, 'wb'))

print()
print("Random Forest Classification Accuracy after Feature Extraction :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()

############################## LOGISTIC REGRESSION ############################

import sklearn
X_new = sklearn.preprocessing.MinMaxScaler().fit_transform(X)

from sklearn.feature_selection import SelectKBest, chi2
n_features = 75
X_new = SelectKBest(chi2, k = n_features).fit_transform(X_new, y)

print("LOGISTIC REGRESSION")
print("Number of Features Selected :", n_features)

# Train-Test Split (70 : 30)
# Stratified Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state = 0, test_size = 0.3,stratify = y)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver = 'liblinear')
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test) 

file = "Weights/LR_Feature_Extraction_Model.sav"
pickle.dump(clf, open(file, 'wb'))

print()
print("Logistic Regression Classification Accuracy after Feature Extraction :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()

############################## K-NEAREST NEIGHBOURS ###########################

import sklearn
X_new = sklearn.preprocessing.MinMaxScaler().fit_transform(X)

from sklearn.feature_selection import SelectKBest, chi2
n_features = 50
X_new = SelectKBest(chi2, k = n_features).fit_transform(X_new, y)

print("K-NEAREST NEIGHBOURS")
print("Number of Features Selected :", n_features)

# Train-Test Split (70 : 30)
# Stratified Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state = 0, test_size = 0.3,stratify = y)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=4)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test) 

file = "Weights/KNN_Feature_Extraction_Model.sav"
pickle.dump(clf, open(file, 'wb'))

print()
print("K-Nearest Neighbors Classification Accuracy after Feature Extraction :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()

############################## DT ###########################

import sklearn
X_new = sklearn.preprocessing.MinMaxScaler().fit_transform(X)

from sklearn.feature_selection import SelectKBest, chi2
n_features = 40
X_new = SelectKBest(chi2, k = n_features).fit_transform(X_new, y)

print("DT")
print("Number of Features Selected :", n_features)

# Train-Test Split (70 : 30)
# Stratified Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state = 0, test_size = 0.3,stratify = y)

from sklearn.datasets import load_iris 
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
file = "Weights/DT_Model.sav"
pickle.dump(clf, open(file, 'wb'))

print()
print("DT Classification Accuracy after Feature Extraction :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()


############################## Multinomial Naive Bayes ###########################

import sklearn
X_new = sklearn.preprocessing.MinMaxScaler().fit_transform(X)

from sklearn.feature_selection import SelectKBest, chi2
n_features = 120
X_new = SelectKBest(chi2, k = n_features).fit_transform(X_new, y)

print("Multinomial Naive Bayes")
print("Number of Features Selected :", n_features)

# Train-Test Split (70 : 30)
# Stratified Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state = 0, test_size = 0.3,stratify = y)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) 

file = "Weights/Multinomial Naive Bayes_Extraction_Model.sav"
pickle.dump(clf, open(file, 'wb'))

print()
print("Multinomial Naive Bayes Classification Accuracy after Feature Extraction :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()


############################## SVC ###########################

import sklearn
X_new = sklearn.preprocessing.MinMaxScaler().fit_transform(X)

from sklearn.feature_selection import SelectKBest, chi2
n_features = 20
X_new = SelectKBest(chi2, k = n_features).fit_transform(X_new, y)

print("SVC")
print("Number of Features Selected :", n_features)

# Train-Test Split (70 : 30)
# Stratified Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state = 0, test_size = 0.3,stratify = y)

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) 

file = "Weights/SVC_Model.sav"
pickle.dump(clf, open(file, 'wb'))

print()
print("SVC Classification Accuracy after Feature Extraction :",( (np.sum(y_pred == y_test)) / len(y_test)) * 100, "%")
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
print()



# sleeping for 1 sec to get 10 sec runtime
time.sleep(1)

# program body ends

# end time
end = time.time()

# total time taken
print()
print(f"Runtime of the program is {(end - start)} seconds")
