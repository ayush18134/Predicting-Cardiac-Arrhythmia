# =============================================================================
# Team - 38
# Adwiteeya Chaudhry (Roll No. 2018126)
# Ayush Bharadwaj (Roll No. 2018134)
# Paras Chaudhary (Roll No. 2018167)
# Indraprashtha Institute of Information Technology, Delhi
# Machine Learning Project
# Predicting Risk of Cardiac Arryhthmia using Machine Learning Techniques
# Data Loading, Preprocessing and Exploratory Data Analysis (EDA)
# =============================================================================

import time

# starting time
start = time.time()

# =============================================================================
#                           IMPORTING LIBRARIES                               #
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (12,8);

# =============================================================================
#                            DATA PREPROCESSING                               #
# =============================================================================

def preprocess(df):
    df.dropna(inplace=True)
    df=np.array(df, dtype=np.str) #converting the array to numpy array
    np.random.seed(0)
    np.random.shuffle(df)

    X=[] #making the input array
    Y=[] #making the input array
    for i in range(len(df)):# appending the data from main file
        x_temp=[]# adding 1 for beta0
        for j in range(len(df[i])-1):
            if(df[i][j]=="?"):
                x_temp.append('0')
            else:
                x_temp.append(df[i][j])
        
        X.append(x_temp)
        # print(x_temp[:12]+x_temp[13:])
        Y.append(df[i][-1])
    # print(X)
    X=np.array(X, dtype=np.float)#converting everything to float
    Y=np.array(Y, dtype=np.float) #converting everything to float
    return X,Y

df = pd.read_csv("Data/Arrhythmia_Dataset.csv")
# df.head()

X,y = preprocess(df)
X_temp,y_temp = preprocess(df)

# Removing the 5 features (out of 279) that contain missing values

df = df.drop(['T','P','QRST','J','Heart Rate'], axis = 1)
df = df.astype('float64')
# df

# Convert the final Class Column into integers (Uniformity of Datatype in the dataframe)

df.iloc[:,-1] = df.iloc[:,-1].astype('int64')

# =============================================================================
#                           EXPLORATORY DATA ANALYSIS                         #
# =============================================================================

# ============================= COUNT PLOT ================================== #

# Tells us the counts of each one of the values present in the column

def class_frequency(y):
    # Frequency Counts
    
    class_list = np.unique(y)
    class_list = class_list.astype('int')
    l = len(class_list)
    
    for i in range(class_list[-1]):
        if i+1 in class_list:
            print("Class", i+1, ":",np.sum(y == i+1))
        else:
            print("Class", i+1, ": 0")
    
    # Plotting the Countplot
    plt.figure();
    ax = sns.countplot(y.astype('int'));
    plt.legend([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
          
    plt.xlabel('Classes') 
    plt.ylabel('Number of Cases') 
    plt.title("Class Counts of Each Class") 
    
    plt.savefig('Plots/Count_Plot.png', format='png')
    plt.show(); 
    
class_frequency(y)   

# ==================== Some measures of Central Tendency ===================== #

print(df.describe())

# ============================== HEAT MAP ==================================== #

plt.figure();
sns.heatmap(df.corr());
 
plt.title("Heat Map") 

plt.savefig('Plots/Heat_Map.png', format='png', bbox_inches = 'tight')
plt.show(); 

# ================================= PCA ====================================== #

## Train Test Splitting

# Train-Test Split (70 : 30)
# Stratified Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3,stratify = y)

# STANDARDIZATION

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

# Fitting the transform on Training Data
scale.fit(X)

# APPLY THE TRANSFORM TO THE ORIGINAL MATRIX

# Transforming 

X_standardized = scale.transform(X)

# Calling TSNE
from sklearn.manifold import TSNE

X_tsne = X_standardized
y_tsne = y

# Dimensions of the space
tsne = TSNE(n_components = 2, random_state=0,perplexity=50.0, n_iter = 10000)

tsne_data = tsne.fit_transform(X_tsne)

tsne_data = np.vstack((tsne_data.T, y_tsne)).T

# Datadframe for input to FacetGrid
tsne_df = pd.DataFrame(data = tsne_data, columns = ("D1","D2","Class"))

# Plotting distribution of the Classes 

plt.figure();
ax = sns.FacetGrid(tsne_df, hue = "Class", height = 6).map(plt.scatter,"D1","D2")
ax.add_legend();
plt.title("tSNE Plot") 

plt.savefig('Plots/tSNE.png', format='png', bbox_inches = 'tight')
plt.show();

# Applying PCA to the training data

from sklearn.decomposition import PCA

pca = PCA(0.95)

# Fit on Training Set only
pca.fit(X_standardized)
print("Number of Features having 95% of the information :",pca.n_components_, "out of", X.shape[1])

# ============================= PCA Plotting ================================= #

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca_comp = pca.fit_transform(X)

pca_comp = np.append(pca_comp, y.reshape(len(y),1), axis = 1)

principal_df = pd.DataFrame(data = pca_comp
             , columns = ['principal component 1', 'principal component 2','CLASS'])
principal_df

class_list = np.unique(y)
class_list = class_list.astype('int')
l = len(class_list)


# Plotting distribution of the Classes 
plt.figure();
ax = sns.FacetGrid(principal_df, hue = "CLASS", height = 8, legend_out=True).map(plt.scatter,"principal component 1","principal component 2")
ax.add_legend()
plt.title("PCA Plot") 

plt.savefig('Plots/PCA_Plot.png', format='png', bbox_inches = 'tight')
plt.show();

# sleeping for 1 sec to get 10 sec runtime
time.sleep(1)

# program body ends

# end time
end = time.time()

# total time taken
print()
print(f"Runtime of the program is {(end - start)} seconds")
