###-------------------------------------- Importing the libraries and packages--------------------------------------###
import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import classification_report as cr
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import scale

from functools import reduce

###-------------------------------------- Importing datasets --------------------------------------###

# Down below, replace put the right paths of the files in your computer
drugbank_mining_category_latest = pd.read_csv('C:\\Users\\Administrateur\\Documents\\CS\\CentraleSupelec\\2018_2019\\Projet inno\\Data\\organized data\\txt\\drugbank_mining_category_latest.txt', sep="\t") 
drugbank_mining_enzymes_latest = pd.read_csv('C:\\Users\\Administrateur\\Documents\\CS\\CentraleSupelec\\2018_2019\\Projet inno\\Data\\organized data\\txt\\drugbank_mining_enzymes_latest.txt', sep="\t") 
drugbank_mining_snp_adverse_drug_reactions_latest = pd.read_csv('C:\\Users\\Administrateur\\Documents\\CS\\CentraleSupelec\\2018_2019\\Projet inno\\Data\\organized data\\txt\\drugbank_mining_snp-adverse-drug-reactions_latest.txt', sep="\t") 
drugbank_mining_targets_latest = pd.read_csv('C:\\Users\\Administrateur\\Documents\\CS\\CentraleSupelec\\2018_2019\\Projet inno\\Data\\organized data\\txt\\drugbank_mining_targets_latest.txt', sep="\t") 
drugbank_mining_transporters_latest = pd.read_csv('C:\\Users\\Administrateur\\Documents\\CS\\CentraleSupelec\\2018_2019\\Projet inno\\Data\\organized data\\txt\\drugbank_mining_transporters_latest.txt', sep="\t")
drugbank_mining_carriers_latest = pd.read_csv('C:\\Users\\Administrateur\\Documents\\CS\\CentraleSupelec\\2018_2019\\Projet inno\\Data\\organized data\\txt\\drugbank_mining_carriers_latest.txt', sep="\t")
drugbank_mining_drug_interactions_latest = pd.read_csv('C:\\Users\\Administrateur\\Documents\\CS\\CentraleSupelec\\2018_2019\\Projet inno\\Data\\organized data\\txt\\drugbank_mining_drug-interactions.latest.txt', sep="\t")
fda= pd.read_csv('C:\\Users\\Administrateur\\Documents\\CS\\CentraleSupelec\\2018_2019\\Projet inno\\Data\\organized data\\fda_variables.csv', sep= ';') 
delay= pd.read_csv('C:\\Users\\Administrateur\\Documents\\CS\\CentraleSupelec\\2018_2019\\Projet inno\\Data\\organized data\\delay.csv', sep=";") 

###-------------------------------------- Feature engineering and cleaning --------------------------------------###
##--Feature cleaning--##

# deleting irrelevant features
fda.pop("DRUG_DEVELOPPED_AS_ANA")
fda.pop("DRUG_DEVELOPPED_AS_ANA_BIN")
fda.dropna( subset=["Number_of_patients_included","Clinical_activity_detected_as following_recurrent_responder_OR_clinical_activity_OR_prolonged_stability", "DLT_identified_or_MTD_reached"], inplace = True)

# Merging the all the features into one dataframe
dataframe=[drugbank_mining_category_latest, drugbank_mining_enzymes_latest, drugbank_mining_snp_adverse_drug_reactions_latest, drugbank_mining_targets_latest,drugbank_mining_transporters_latest, drugbank_mining_carriers_latest,drugbank_mining_drug_interactions_latest]

txt_merged=reduce((lambda x,y: pd.merge(x,y, on ="Unnamed: 0" )), dataframe)
txt_merged.rename(columns={'Unnamed: 0':'Drug'}, inplace=True)
full_labeled_dataset_df= pd.merge(txt_merged,fda, on ="Drug", sort=True)

delay.rename(columns={'COMMON_DRUGBANK_ALIAS':'Drug'}, inplace= True)
full_labeled_dataset_df= pd.merge(full_labeled_dataset_df,delay, on ="Drug", sort=True)


##--Data Engineering--##
# Defining a function that modifies the target depending on the delay of approval.
def act(Data_df,Seuil):
    Data=Data_df
    Data=Data.reset_index(drop=True)
    labels_df = Data['FDA_APPROVAL']
    delai = Data['DELAY_FROM_OLDEST_PMID_TO_FDA_APPROVAL_or_DDN']
    drop=[]
    for i in range(len(labels_df)):
        if delai.iloc[i] > Seuil :
            labels_df.iloc[i] = 0
    
    drop=[i for i in range(len(labels_df)) if delai.iloc[i] < Seuil and labels_df.iloc[i] == 0]
    
    Data.drop(drop,inplace=True)
    labels_df.drop(drop, inplace=True)
    Data.drop(['DELAY_FROM_OLDEST_PMID_TO_FDA_APPROVAL_or_DDN', 'FDA_APPROVAL', 'Drug'], axis=1, inplace=True)
    full_dataset=Data.values
    labels=labels_df.values
    X_train, X_test, Y_train, Y_test = train_test_split(full_dataset, labels, test_size=0.33, random_state=42)
    
    return X_train, X_test, Y_train, Y_test


##--Feature Selection--##
# Doing feature selection by searching the most important features thanks to Random Forest algorithm 
# For that we first search the best hyperparameters for predicting approvals for random forest thanks to Grid Search algorithm
def rfc_param_selection(X, y, nfolds):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # Implemeting the grid search
    rfc=RandomForestClassifier()
    rfc_gridsearch= GridSearchCV(estimator = rfc, param_grid = random_grid, cv =nfolds , n_jobs = -1, verbose = 2)
    rfc_gridsearch.fit(X, y)
    return rfc_gridsearch.best_params_



# Splitting the full dataset into training and testing sets
# To test another period change the numerical value in the funtion act() below
X_train, X_test, Y_train, Y_test = act(full_labeled_dataset_df, 180)

# fitting grid search for getting the most important features
rfc_best_params= rfc_param_selection(X_train, Y_train, 3)
rfc_best_params_grid=RandomForestClassifier(n_estimators=rfc_best_params['n_estimators'], max_depth=rfc_best_params['max_depth'], min_samples_split=rfc_best_params['min_samples_split'], min_samples_leaf=rfc_best_params['min_samples_leaf'], max_features=rfc_best_params['max_features'], bootstrap=rfc_best_params['bootstrap'], random_state=42)
rfc_best_params_grid.fit(X_train, Y_train)

# Get numerical feature importances
importances = list(rfc_best_params_grid.feature_importances_)
feature_list=[e for e in full_labeled_dataset_df.columns.tolist() if e not in ['DELAY_FROM_OLDEST_PMID_TO_FDA_APPROVAL_or_DDN', 'FDA_APPROVAL', 'Drug']]

# List of tuples with variable and importance
feature_importances = [(feature, importance) for feature, importance in zip(feature_list, importances)]
feature_importances_dict= dict(feature_importances)
indexes_important= tuple([feature_list.index(e) for e in feature_list if feature_importances_dict[e]>0.0 ])

# Retrieving the most important features
X_train_important= X_train[:,indexes_important]
X_test_important= X_test[:,indexes_important]

###-------------------------------------------------- Deploying the models ---------------------------------------------###

## Random Forest Classifier

# In[5]:


accuracy_train_rfc=[]
rocAuc_train_rfc=[]
precision_train_rfc=[]
recall_train_rfc=[]

accuracy_test_rfc=[]
rocAuc_test_rfc=[]
precision_test_rfc=[]
recall_test_rfc=[]


# In[6]:


# rfc raw for demonstration purpose
rfc_raw=RandomForestClassifier()
rfc_raw.fit(X_train, Y_train)
Y_train_rfc_raw = rfc_raw.predict(X_train)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_rfc_raw))
print("Classification report for the training set: \n", cr(Y_train, Y_train_rfc_raw))
accuracy_train_rfc.append(accuracy_score(Y_train, Y_train_rfc_raw))
rocAuc_train_rfc.append(roc_auc_score(Y_train, Y_train_rfc_raw))
precision_train_rfc.append(precision_score(Y_train, Y_train_rfc_raw))
recall_train_rfc.append(recall_score(Y_train, Y_train_rfc_raw))


Y_test_rfc_raw=rfc_raw.predict(X_test)
accuracy_test_rfc.append(accuracy_score(Y_test, Y_test_rfc_raw))
rocAuc_test_rfc.append(roc_auc_score(Y_test, Y_test_rfc_raw))
precision_test_rfc.append(precision_score(Y_test, Y_test_rfc_raw))
recall_test_rfc.append(recall_score(Y_test, Y_test_rfc_raw))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_rfc_raw))
cm= confusion_matrix(Y_test, Y_test_rfc_raw)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_rfc_raw))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_rfc_raw))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_rfc_raw))



# In[7]:


# Random Forest Classifier with the best parameters and most important features
rfc=RandomForestClassifier(n_estimators=rfc_best_params['n_estimators'], max_depth=rfc_best_params['max_depth'], min_samples_split=rfc_best_params['min_samples_split'], min_samples_leaf=rfc_best_params['min_samples_leaf'], max_features=rfc_best_params['max_features'], bootstrap=rfc_best_params['bootstrap'], random_state=42)
rfc.fit(X_train_important, Y_train)
Y_train_rfc = rfc.predict(X_train_important)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_rfc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_rfc))
accuracy_train_rfc.append(accuracy_score(Y_train, Y_train_rfc))
rocAuc_train_rfc.append(roc_auc_score(Y_train, Y_train_rfc))
precision_train_rfc.append(precision_score(Y_train, Y_train_rfc))
recall_train_rfc.append(recall_score(Y_train, Y_train_rfc))

Y_test_rfc=rfc.predict(X_test_important)
accuracy_test_rfc.append(accuracy_score(Y_test, Y_test_rfc))
rocAuc_test_rfc.append(roc_auc_score(Y_test, Y_test_rfc))
precision_test_rfc.append(precision_score(Y_test, Y_test_rfc))
recall_test_rfc.append(recall_score(Y_test, Y_test_rfc))
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_rfc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_rfc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_rfc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_rfc))


# In[8]:


# Random Forest Classifier with the best parameters and most important features scaled
rfc=RandomForestClassifier(n_estimators=rfc_best_params['n_estimators'], max_depth=rfc_best_params['max_depth'], min_samples_split=rfc_best_params['min_samples_split'], min_samples_leaf=rfc_best_params['min_samples_leaf'], max_features=rfc_best_params['max_features'], bootstrap=rfc_best_params['bootstrap'], random_state=42)
rfc.fit(scale(X_train_important), Y_train)
Y_train_rfc = rfc.predict(scale(X_train_important))
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_rfc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_rfc))
accuracy_train_rfc.append(accuracy_score(Y_train, Y_train_rfc))
rocAuc_train_rfc.append(roc_auc_score(Y_train, Y_train_rfc))
precision_train_rfc.append(precision_score(Y_train, Y_train_rfc))
recall_train_rfc.append(recall_score(Y_train, Y_train_rfc))

Y_test_rfc=rfc.predict(scale(X_test_important))
accuracy_test_rfc.append(accuracy_score(Y_test, Y_test_rfc))
rocAuc_test_rfc.append(roc_auc_score(Y_test, Y_test_rfc))
precision_test_rfc.append(precision_score(Y_test, Y_test_rfc))
recall_test_rfc.append(recall_score(Y_test, Y_test_rfc))
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_rfc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_rfc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_rfc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_rfc))


# In[9]:


# Random Forest Classifier with the best parameters and all the features
rfc=RandomForestClassifier(n_estimators=rfc_best_params['n_estimators'], max_depth=rfc_best_params['max_depth'], min_samples_split=rfc_best_params['min_samples_split'], min_samples_leaf=rfc_best_params['min_samples_leaf'], max_features=rfc_best_params['max_features'], bootstrap=rfc_best_params['bootstrap'], random_state=42)
rfc.fit(X_train, Y_train)
Y_train_rfc = rfc.predict(X_train)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_rfc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_rfc))
accuracy_train_rfc.append(accuracy_score(Y_train, Y_train_rfc))
rocAuc_train_rfc.append(roc_auc_score(Y_train, Y_train_rfc))
precision_train_rfc.append(precision_score(Y_train, Y_train_rfc))
recall_train_rfc.append(recall_score(Y_train, Y_train_rfc))

Y_test_rfc=rfc.predict(X_test)
accuracy_test_rfc.append(accuracy_score(Y_test, Y_test_rfc))
rocAuc_test_rfc.append(roc_auc_score(Y_test, Y_test_rfc))
precision_test_rfc.append(precision_score(Y_test, Y_test_rfc))
recall_test_rfc.append(recall_score(Y_test, Y_test_rfc))
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_rfc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_rfc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_rfc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_rfc))


# In[10]:


# Random Forest Classifier with the best parameters and all the features scaled
rfc=RandomForestClassifier(n_estimators=rfc_best_params['n_estimators'], max_depth=rfc_best_params['max_depth'], min_samples_split=rfc_best_params['min_samples_split'], min_samples_leaf=rfc_best_params['min_samples_leaf'], max_features=rfc_best_params['max_features'], bootstrap=rfc_best_params['bootstrap'], random_state=42)
rfc.fit(scale(X_train), Y_train)
Y_train_rfc = rfc.predict(scale(X_train))
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_rfc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_rfc))
accuracy_train_rfc.append(accuracy_score(Y_train, Y_train_rfc))
rocAuc_train_rfc.append(roc_auc_score(Y_train, Y_train_rfc))
precision_train_rfc.append(precision_score(Y_train, Y_train_rfc))
recall_train_rfc.append(recall_score(Y_train, Y_train_rfc))

Y_test_rfc=rfc.predict(scale(X_test))
accuracy_test_rfc.append(accuracy_score(Y_test, Y_test_rfc))
rocAuc_test_rfc.append(roc_auc_score(Y_test, Y_test_rfc))
precision_test_rfc.append(precision_score(Y_test, Y_test_rfc))
recall_test_rfc.append(recall_score(Y_test, Y_test_rfc))
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_rfc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_rfc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_rfc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_rfc))


## Support Vector Machine

## Linear Kernel

# In[11]:


accuracy_train_linear_svc=[]
rocAuc_train_linear_svc=[]
precision_train_linear_svc=[]
recall_train_linear_svc=[]

accuracy_test_linear_svc=[]
rocAuc_test_linear_svc=[]
precision_test_linear_svc=[]
recall_test_linear_svc=[]


# In[12]:


def linear_svc_param_selection(X, y, nfolds):
    C= [0.001, 0.01, 0.1, 1, 10, 25, 50]
    param_grid = {'C': C}
    grid_search = GridSearchCV(svm.SVC(kernel='linear',random_state=42), param_grid=param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

best_C = linear_svc_param_selection(scale(X_train_important), Y_train, 3)


# In[13]:


# linear kernel raw for demonstration purpose
linear_svc=svm.SVC(kernel='linear', C=best_C['C'], random_state=42)
linear_svc.fit(X_train, Y_train)

Y_train_linear_svc = linear_svc.predict(X_train)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_linear_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_linear_svc))
accuracy_train_linear_svc.append(accuracy_score(Y_train, Y_train_linear_svc))
rocAuc_train_linear_svc.append(roc_auc_score(Y_train, Y_train_linear_svc))
precision_train_linear_svc.append(precision_score(Y_train, Y_train_linear_svc))
recall_train_linear_svc.append(recall_score(Y_train, Y_train_linear_svc))

Y_test_linear_svc=linear_svc.predict(X_test)
accuracy_test_linear_svc.append(accuracy_score(Y_test, Y_test_linear_svc))
rocAuc_test_linear_svc.append(roc_auc_score(Y_test, Y_test_linear_svc))
precision_test_linear_svc.append(precision_score(Y_test, Y_test_linear_svc))
recall_test_linear_svc.append(recall_score(Y_test, Y_test_linear_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_linear_svc))
cm= confusion_matrix(Y_test, Y_test_linear_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_linear_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_linear_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_linear_svc))

# In[14]:


# linear kernel classifier with the best parameters and most important features
linear_svc=svm.SVC(kernel='linear', C=best_C['C'], random_state=42)
linear_svc.fit(X_train_important, Y_train)
Y_train_linear_svc = linear_svc.predict(X_train_important)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_linear_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_linear_svc))
accuracy_train_linear_svc.append(accuracy_score(Y_train, Y_train_linear_svc))
rocAuc_train_linear_svc.append(roc_auc_score(Y_train, Y_train_linear_svc))
precision_train_linear_svc.append(precision_score(Y_train, Y_train_linear_svc))
recall_train_linear_svc.append(recall_score(Y_train, Y_train_linear_svc))

Y_test_linear_svc=linear_svc.predict(X_test_important)
accuracy_test_linear_svc.append(accuracy_score(Y_test, Y_test_linear_svc))
rocAuc_test_linear_svc.append(roc_auc_score(Y_test, Y_test_linear_svc))
precision_test_linear_svc.append(precision_score(Y_test, Y_test_linear_svc))
recall_test_linear_svc.append(recall_score(Y_test, Y_test_linear_svc))
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_linear_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_linear_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_linear_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_linear_svc))


# In[15]:


# linear kernel with the best parameters and most important features scaled
linear_svc=svm.SVC(kernel='linear', C=best_C['C'], random_state=42)
linear_svc.fit(scale(X_train_important), Y_train)
Y_train_linear_svc = linear_svc.predict(scale(X_train_important))
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_linear_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_linear_svc))
accuracy_train_linear_svc.append(accuracy_score(Y_train, Y_train_linear_svc))
rocAuc_train_linear_svc.append(roc_auc_score(Y_train, Y_train_linear_svc))
precision_train_linear_svc.append(precision_score(Y_train, Y_train_linear_svc))
recall_train_linear_svc.append(recall_score(Y_train, Y_train_linear_svc))

Y_test_linear_svc=linear_svc.predict(scale(X_test_important))
accuracy_test_linear_svc.append(accuracy_score(Y_test, Y_test_linear_svc))
rocAuc_test_linear_svc.append(roc_auc_score(Y_test, Y_test_linear_svc))
precision_test_linear_svc.append(precision_score(Y_test, Y_test_linear_svc))
recall_test_linear_svc.append(recall_score(Y_test, Y_test_linear_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_linear_svc))
cm= confusion_matrix(Y_test, Y_test_linear_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_linear_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_linear_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_linear_svc))


# In[16]:


# linear kernel with the best parameters and all the features
linear_svc=svm.SVC(kernel='linear', C=best_C['C'], random_state=42)
linear_svc.fit(X_train, Y_train)
Y_train_linear_svc = linear_svc.predict(X_train)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_linear_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_linear_svc))
accuracy_train_linear_svc.append(accuracy_score(Y_train, Y_train_linear_svc))
rocAuc_train_linear_svc.append(roc_auc_score(Y_train, Y_train_linear_svc))
precision_train_linear_svc.append(precision_score(Y_train, Y_train_linear_svc))
recall_train_linear_svc.append(recall_score(Y_train, Y_train_linear_svc))

Y_test_linear_svc=linear_svc.predict(X_test)
accuracy_test_linear_svc.append(accuracy_score(Y_test, Y_test_linear_svc))
rocAuc_test_linear_svc.append(roc_auc_score(Y_test, Y_test_linear_svc))
precision_test_linear_svc.append(precision_score(Y_test, Y_test_linear_svc))
recall_test_linear_svc.append(recall_score(Y_test, Y_test_linear_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_linear_svc))
cm= confusion_matrix(Y_test, Y_test_linear_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_linear_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_linear_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_linear_svc))


# In[17]:


# linear kernel with the best parameters and all features scaled
linear_svc=svm.SVC(kernel='linear', C=best_C['C'], random_state=42)
linear_svc.fit(scale(X_train), Y_train)
Y_train_linear_svc = linear_svc.predict(scale(X_train))
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_linear_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_linear_svc))
accuracy_train_linear_svc.append(accuracy_score(Y_train, Y_train_linear_svc))
rocAuc_train_linear_svc.append(roc_auc_score(Y_train, Y_train_linear_svc))
precision_train_linear_svc.append(precision_score(Y_train, Y_train_linear_svc))
recall_train_linear_svc.append(recall_score(Y_train, Y_train_linear_svc))

Y_test_linear_svc=linear_svc.predict(scale(X_test))
accuracy_test_linear_svc.append(accuracy_score(Y_test, Y_test_linear_svc))
rocAuc_test_linear_svc.append(roc_auc_score(Y_test, Y_test_linear_svc))
precision_test_linear_svc.append(precision_score(Y_test, Y_test_linear_svc))
recall_test_linear_svc.append(recall_score(Y_test, Y_test_linear_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_linear_svc))
cm= confusion_matrix(Y_test, Y_test_linear_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_linear_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_linear_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_linear_svc))


## RBF KERNEL

# In[18]:


accuracy_train_rbf=[]
rocAuc_train_rbf=[]
precision_train_rbf=[]
recall_train_rbf=[]

accuracy_test_rbf=[]
rocAuc_test_rbf=[]
precision_test_rbf=[]
recall_test_rbf=[]


# In[19]:


def rbf_svc_param_selection(X, y, nfolds):
    C= [0.001, 0.01, 0.1, 1, 10, 25, 50]
    param_grid = {'gamma': [0.0001,0.001, 0.01, 0.1, 1],
                     'C': [0.001, 0.01, 0.1, 1, 10]}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf',random_state=42), param_grid=param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

best_params_rbf = rbf_svc_param_selection(scale(X_train_important), Y_train, 3)


# In[20]:


# Rbf kernel raw for demonstration purpose
rbf_svc=svm.SVC(kernel='rbf', random_state=42)
rbf_svc.fit(X_train, Y_train)
Y_train_rbf_svc = rbf_svc.predict(X_train)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_rbf_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_rbf_svc))
accuracy_train_rbf.append(accuracy_score(Y_train, Y_train_rbf_svc))
rocAuc_train_rbf.append(roc_auc_score(Y_train, Y_train_rbf_svc))
precision_train_rbf.append(precision_score(Y_train, Y_train_rbf_svc))
recall_train_rbf.append(recall_score(Y_train, Y_train_rbf_svc))

Y_test_rbf_svc=rbf_svc.predict(X_test)
accuracy_test_rbf.append(accuracy_score(Y_test, Y_test_rbf_svc))
rocAuc_test_rbf.append(roc_auc_score(Y_test, Y_test_rbf_svc))
precision_test_rbf.append(precision_score(Y_test, Y_test_rbf_svc))
recall_test_rbf.append(recall_score(Y_test, Y_test_rbf_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_rbf_svc))
cm= confusion_matrix(Y_test, Y_test_rbf_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_rbf_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_rbf_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_rbf_svc))


# In[21]:


# Rbf kernel classifier with the best parameters and most important features
rbf_svc=svm.SVC(kernel='rbf', C=best_params_rbf['C'], gamma= best_params_rbf['gamma'],random_state=42)
rbf_svc.fit(X_train_important, Y_train)
Y_train_rbf_svc = rbf_svc.predict(X_train_important)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_rbf_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_rbf_svc))
accuracy_train_rbf.append(accuracy_score(Y_train, Y_train_rbf_svc))
rocAuc_train_rbf.append(roc_auc_score(Y_train, Y_train_rbf_svc))
precision_train_rbf.append(precision_score(Y_train, Y_train_rbf_svc))
recall_train_rbf.append(recall_score(Y_train, Y_train_rbf_svc))

Y_test_rbf_svc=rbf_svc.predict(X_test_important)
accuracy_test_rbf.append(accuracy_score(Y_test, Y_test_rbf_svc))
rocAuc_test_rbf.append(roc_auc_score(Y_test, Y_test_rbf_svc))
precision_test_rbf.append(precision_score(Y_test, Y_test_rbf_svc))
recall_test_rbf.append(recall_score(Y_test, Y_test_rbf_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_rbf_svc))
cm= confusion_matrix(Y_test, Y_test_rbf_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_rbf_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_rbf_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_rbf_svc))


# In[22]:


# Rbf kernel classifier with the best parameters and most important features scaled
rbf_svc=svm.SVC(kernel='rbf', C=best_params_rbf['C'], gamma= best_params_rbf['gamma'],random_state=42)
rbf_svc.fit(scale(X_train_important), Y_train)
Y_train_rbf_svc = rbf_svc.predict(scale(X_train_important))
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_rbf_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_rbf_svc))
accuracy_train_rbf.append(accuracy_score(Y_train, Y_train_rbf_svc))
rocAuc_train_rbf.append(roc_auc_score(Y_train, Y_train_rbf_svc))
precision_train_rbf.append(precision_score(Y_train, Y_train_rbf_svc))
recall_train_rbf.append(recall_score(Y_train, Y_train_rbf_svc))

Y_test_rbf_svc=rbf_svc.predict(scale(X_test_important))
accuracy_test_rbf.append(accuracy_score(Y_test, Y_test_rbf_svc))
rocAuc_test_rbf.append(roc_auc_score(Y_test, Y_test_rbf_svc))
precision_test_rbf.append(precision_score(Y_test, Y_test_rbf_svc))
recall_test_rbf.append(recall_score(Y_test, Y_test_rbf_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_rbf_svc))
cm= confusion_matrix(Y_test, Y_test_rbf_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_rbf_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_rbf_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_rbf_svc))


# In[23]:


# Rbf kernel classifier with the best parameters and all features
rbf_svc=svm.SVC(kernel='rbf', C=best_params_rbf['C'], gamma= best_params_rbf['gamma'],random_state=42)
rbf_svc.fit(X_train, Y_train)
Y_train_rbf_svc = rbf_svc.predict(X_train)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_rbf_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_rbf_svc))
accuracy_train_rbf.append(accuracy_score(Y_train, Y_train_rbf_svc))
rocAuc_train_rbf.append(roc_auc_score(Y_train, Y_train_rbf_svc))
precision_train_rbf.append(precision_score(Y_train, Y_train_rbf_svc))
recall_train_rbf.append(recall_score(Y_train, Y_train_rbf_svc))

Y_test_rbf_svc=rbf_svc.predict(X_test)
accuracy_test_rbf.append(accuracy_score(Y_test, Y_test_rbf_svc))
rocAuc_test_rbf.append(roc_auc_score(Y_test, Y_test_rbf_svc))
precision_test_rbf.append(precision_score(Y_test, Y_test_rbf_svc))
recall_test_rbf.append(recall_score(Y_test, Y_test_rbf_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_rbf_svc))
cm= confusion_matrix(Y_test, Y_test_rbf_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_rbf_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_rbf_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_rbf_svc))


# In[24]:


# Rbf kernel classifier with the best parameters and all features scaled
rbf_svc=svm.SVC(kernel='rbf', C=best_params_rbf['C'], gamma= best_params_rbf['gamma'],random_state=42)
rbf_svc.fit(scale(X_train), Y_train)
Y_train_rbf_svc = rbf_svc.predict(scale(X_train))
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_rbf_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_rbf_svc))
accuracy_train_rbf.append(accuracy_score(Y_train, Y_train_rbf_svc))
rocAuc_train_rbf.append(roc_auc_score(Y_train, Y_train_rbf_svc))
precision_train_rbf.append(precision_score(Y_train, Y_train_rbf_svc))
recall_train_rbf.append(recall_score(Y_train, Y_train_rbf_svc))

Y_test_rbf_svc=rbf_svc.predict(scale(X_test))
accuracy_test_rbf.append(accuracy_score(Y_test, Y_test_rbf_svc))
rocAuc_test_rbf.append(roc_auc_score(Y_test, Y_test_rbf_svc))
precision_test_rbf.append(precision_score(Y_test, Y_test_rbf_svc))
recall_test_rbf.append(recall_score(Y_test, Y_test_rbf_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_rbf_svc))
cm= confusion_matrix(Y_test, Y_test_rbf_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_rbf_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_rbf_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_rbf_svc))


## Polynomial Kernel

# In[25]:


accuracy_train_poly=[]
rocAuc_train_poly=[]
precision_train_poly=[]
recall_train_poly=[]

accuracy_test_poly=[]
rocAuc_test_poly=[]
precision_test_poly=[]
recall_test_poly=[]


# In[26]:


def poly_svc_param_selection(X, y, nfolds):
    param_grid = {'gamma': [0.0001,0.001, 0.01, 0.1, 1],
                     'C': [0.001, 0.01, 0.1, 1, 10],
                 'degree': [0, 1, 2, 3, 4, 5]}
    grid_search = GridSearchCV(svm.SVC(kernel='poly',random_state=42), param_grid=param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

best_params_poly = poly_svc_param_selection(scale(X_train_important), Y_train, 3)


# In[27]:


# poly kernel raw for demonstration purpose
poly_svc=svm.SVC(kernel='poly', random_state=42)
poly_svc.fit(X_train, Y_train)
Y_train_poly_svc = poly_svc.predict(X_train)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_poly_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_poly_svc))
accuracy_train_poly.append(accuracy_score(Y_train, Y_train_poly_svc))
rocAuc_train_poly.append(roc_auc_score(Y_train, Y_train_poly_svc))
precision_train_poly.append(precision_score(Y_train, Y_train_poly_svc))
recall_train_poly.append(recall_score(Y_train, Y_train_poly_svc))

Y_test_poly_svc=poly_svc.predict(X_test)
accuracy_test_poly.append(accuracy_score(Y_test, Y_test_poly_svc))
rocAuc_test_poly.append(roc_auc_score(Y_test, Y_test_poly_svc))
precision_test_poly.append(precision_score(Y_test, Y_test_poly_svc))
recall_test_poly.append(recall_score(Y_test, Y_test_poly_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_poly_svc))
cm= confusion_matrix(Y_test, Y_test_poly_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_poly_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_poly_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_poly_svc))


# In[28]:


# poly kernel classifier with the best parameters and most important features
poly_svc=svm.SVC(kernel='poly', C=best_params_poly['C'], gamma= best_params_poly['gamma'], degree=best_params_poly['degree'], random_state=42)
poly_svc.fit(X_train_important, Y_train)
Y_train_poly_svc = poly_svc.predict(X_train_important)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_poly_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_poly_svc))
accuracy_train_poly.append(accuracy_score(Y_train, Y_train_poly_svc))
rocAuc_train_poly.append(roc_auc_score(Y_train, Y_train_poly_svc))
precision_train_poly.append(precision_score(Y_train, Y_train_poly_svc))
recall_train_poly.append(recall_score(Y_train, Y_train_poly_svc))

Y_test_poly_svc=poly_svc.predict(X_test_important)
accuracy_test_poly.append(accuracy_score(Y_test, Y_test_poly_svc))
rocAuc_test_poly.append(roc_auc_score(Y_test, Y_test_poly_svc))
precision_test_poly.append(precision_score(Y_test, Y_test_poly_svc))
recall_test_poly.append(recall_score(Y_test, Y_test_poly_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_poly_svc))
cm= confusion_matrix(Y_test, Y_test_poly_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_poly_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_poly_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_poly_svc))


# In[29]:


# poly kernel classifier with the best parameters and most important features scaled
poly_svc=svm.SVC(kernel='poly', C=best_params_poly['C'], gamma= best_params_poly['gamma'], degree=best_params_poly['degree'], random_state=42)
poly_svc.fit(scale(X_train_important), Y_train)
Y_train_poly_svc = poly_svc.predict(scale(X_train_important))
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_poly_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_poly_svc))
accuracy_train_poly.append(accuracy_score(Y_train, Y_train_poly_svc))
rocAuc_train_poly.append(roc_auc_score(Y_train, Y_train_poly_svc))
precision_train_poly.append(precision_score(Y_train, Y_train_poly_svc))
recall_train_poly.append(recall_score(Y_train, Y_train_poly_svc))

Y_test_poly_svc=poly_svc.predict(scale(X_test_important))
accuracy_test_poly.append(accuracy_score(Y_test, Y_test_poly_svc))
rocAuc_test_poly.append(roc_auc_score(Y_test, Y_test_poly_svc))
precision_test_poly.append(precision_score(Y_test, Y_test_poly_svc))
recall_test_poly.append(recall_score(Y_test, Y_test_poly_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_poly_svc))
cm= confusion_matrix(Y_test, Y_test_poly_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_poly_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_poly_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_poly_svc))


# In[30]:


# poly kernel classifier with the best parameters and all features
poly_svc=svm.SVC(kernel='poly', C=best_params_poly['C'], gamma= best_params_poly['gamma'], degree=best_params_poly['degree'], random_state=42)
poly_svc.fit(X_train, Y_train)
Y_train_poly_svc = poly_svc.predict(X_train)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_poly_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_poly_svc))
accuracy_train_poly.append(accuracy_score(Y_train, Y_train_poly_svc))
rocAuc_train_poly.append(roc_auc_score(Y_train, Y_train_poly_svc))
precision_train_poly.append(precision_score(Y_train, Y_train_poly_svc))
recall_train_poly.append(recall_score(Y_train, Y_train_poly_svc))

Y_test_poly_svc=poly_svc.predict(X_test)
accuracy_test_poly.append(accuracy_score(Y_test, Y_test_poly_svc))
rocAuc_test_poly.append(roc_auc_score(Y_test, Y_test_poly_svc))
precision_test_poly.append(precision_score(Y_test, Y_test_poly_svc))
recall_test_poly.append(recall_score(Y_test, Y_test_poly_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_poly_svc))
cm= confusion_matrix(Y_test, Y_test_poly_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_poly_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_poly_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_poly_svc))


# In[31]:


# poly kernel classifier with the best parameters and all features scaled
poly_svc=svm.SVC(kernel='poly', C=best_params_poly['C'], gamma= best_params_poly['gamma'], degree=best_params_poly['degree'], random_state=42)
poly_svc.fit(scale(X_train), Y_train)
Y_train_poly_svc = poly_svc.predict(scale(X_train))
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_poly_svc))
print("Classification report for the training set: \n", cr(Y_train, Y_train_poly_svc))
accuracy_train_poly.append(accuracy_score(Y_train, Y_train_poly_svc))
rocAuc_train_poly.append(roc_auc_score(Y_train, Y_train_poly_svc))
precision_train_poly.append(precision_score(Y_train, Y_train_poly_svc))
recall_train_poly.append(recall_score(Y_train, Y_train_poly_svc))

Y_test_poly_svc=poly_svc.predict(scale(X_test))
accuracy_test_poly.append(accuracy_score(Y_test, Y_test_poly_svc))
rocAuc_test_poly.append(roc_auc_score(Y_test, Y_test_poly_svc))
precision_test_poly.append(precision_score(Y_test, Y_test_poly_svc))
recall_test_poly.append(recall_score(Y_test, Y_test_poly_svc))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_poly_svc))
cm= confusion_matrix(Y_test, Y_test_poly_svc)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_poly_svc))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_poly_svc))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_poly_svc))


## Logistic Regression

# In[32]:


accuracy_train_logReg=[]
rocAuc_train_logReg=[]
precision_train_logReg=[]
recall_train_logReg=[]

accuracy_test_logReg=[]
rocAuc_test_logReg=[]
precision_test_logReg=[]
recall_test_logReg=[]


# In[33]:


def logReg_param_selection(X, y, nfolds):
    C= [0.001, 0.01, 0.1, 1, 10, 25, 50]
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty':['l2', 'l1'] }
    grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid=param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

best_params_logReg = logReg_param_selection(scale(X_train_important), Y_train, 3)


# In[34]:


# Log Reg raw for demonstration purpose
logReg=LogisticRegression(random_state=42)
logReg.fit(X_train, Y_train)
Y_train_logReg = logReg.predict(X_train)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_logReg))
print("Classification report for the training set: \n", cr(Y_train, Y_train_logReg))
accuracy_train_logReg.append(accuracy_score(Y_train, Y_train_logReg))
rocAuc_train_logReg.append(roc_auc_score(Y_train, Y_train_logReg))
precision_train_logReg.append(precision_score(Y_train, Y_train_logReg))
recall_train_logReg.append(recall_score(Y_train, Y_train_logReg))

Y_test_logReg=logReg.predict(X_test)
accuracy_test_logReg.append(accuracy_score(Y_test, Y_test_logReg))
rocAuc_test_logReg.append(roc_auc_score(Y_test, Y_test_logReg))
precision_test_logReg.append(precision_score(Y_test, Y_test_logReg))
recall_test_logReg.append(recall_score(Y_test, Y_test_logReg))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_logReg))
cm= confusion_matrix(Y_test, Y_test_logReg)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_logReg))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_logReg))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_logReg))


# In[35]:


# Log Reg with the best parameters and most important features
logReg=LogisticRegression(random_state=42, penalty=best_params_logReg['penalty'], C=best_params_logReg['C'])
logReg.fit(X_train_important, Y_train)
Y_train_logReg = logReg.predict(X_train_important)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_logReg))
print("Classification report for the training set: \n", cr(Y_train, Y_train_logReg))
accuracy_train_logReg.append(accuracy_score(Y_train, Y_train_logReg))
rocAuc_train_logReg.append(roc_auc_score(Y_train, Y_train_logReg))
precision_train_logReg.append(precision_score(Y_train, Y_train_logReg))
recall_train_logReg.append(recall_score(Y_train, Y_train_logReg))

Y_test_logReg=logReg.predict(X_test_important)
accuracy_test_logReg.append(accuracy_score(Y_test, Y_test_logReg))
rocAuc_test_logReg.append(roc_auc_score(Y_test, Y_test_logReg))
precision_test_logReg.append(precision_score(Y_test, Y_test_logReg))
recall_test_logReg.append(recall_score(Y_test, Y_test_logReg))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_logReg))
cm= confusion_matrix(Y_test, Y_test_logReg)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_logReg))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_logReg))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_logReg))


# In[36]:


# Log Reg with the best parameters and most important features scaled
logReg=LogisticRegression(random_state=42, penalty=best_params_logReg['penalty'], C=best_params_logReg['C'])
logReg.fit(scale(X_train_important), Y_train)
Y_train_logReg = logReg.predict(scale(X_train_important))
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_logReg))
print("Classification report for the training set: \n", cr(Y_train, Y_train_logReg))
accuracy_train_logReg.append(accuracy_score(Y_train, Y_train_logReg))
rocAuc_train_logReg.append(roc_auc_score(Y_train, Y_train_logReg))
precision_train_logReg.append(precision_score(Y_train, Y_train_logReg))
recall_train_logReg.append(recall_score(Y_train, Y_train_logReg))

Y_test_logReg=logReg.predict(scale(X_test_important))
accuracy_test_logReg.append(accuracy_score(Y_test, Y_test_logReg))
rocAuc_test_logReg.append(roc_auc_score(Y_test, Y_test_logReg))
precision_test_logReg.append(precision_score(Y_test, Y_test_logReg))
recall_test_logReg.append(recall_score(Y_test, Y_test_logReg))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_logReg))
cm= confusion_matrix(Y_test, Y_test_logReg)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_logReg))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_logReg))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_logReg))


# In[37]:


# Log Reg with the best parameters and all the features
logReg=LogisticRegression(random_state=42, penalty=best_params_logReg['penalty'], C=best_params_logReg['C'])
logReg.fit(X_train, Y_train)
Y_train_logReg = logReg.predict(X_train)
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_logReg))
print("Classification report for the training set: \n", cr(Y_train, Y_train_logReg))
accuracy_train_logReg.append(accuracy_score(Y_train, Y_train_logReg))
rocAuc_train_logReg.append(roc_auc_score(Y_train, Y_train_logReg))
precision_train_logReg.append(precision_score(Y_train, Y_train_logReg))
recall_train_logReg.append(recall_score(Y_train, Y_train_logReg))

Y_test_logReg=logReg.predict(X_test)
accuracy_test_logReg.append(accuracy_score(Y_test, Y_test_logReg))
rocAuc_test_logReg.append(roc_auc_score(Y_test, Y_test_logReg))
precision_test_logReg.append(precision_score(Y_test, Y_test_logReg))
recall_test_logReg.append(recall_score(Y_test, Y_test_logReg))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_logReg))
cm= confusion_matrix(Y_test, Y_test_logReg)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_logReg))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_logReg))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_logReg))


# In[38]:


# Log Reg with the best parameters and all features scaled
logReg=LogisticRegression(random_state=42, penalty=best_params_logReg['penalty'], C=best_params_logReg['C'])
logReg.fit(scale(X_train), Y_train)
Y_train_logReg = logReg.predict(scale(X_train))
print("Confusion matrix for the training set: \n", confusion_matrix(Y_train, Y_train_logReg))
print("Classification report for the training set: \n", cr(Y_train, Y_train_logReg))
accuracy_train_logReg.append(accuracy_score(Y_train, Y_train_logReg))
rocAuc_train_logReg.append(roc_auc_score(Y_train, Y_train_logReg))
precision_train_logReg.append(precision_score(Y_train, Y_train_logReg))
recall_train_logReg.append(recall_score(Y_train, Y_train_logReg))

Y_test_logReg=logReg.predict(scale(X_test))
accuracy_test_logReg.append(accuracy_score(Y_test, Y_test_logReg))
rocAuc_test_logReg.append(roc_auc_score(Y_test, Y_test_logReg))
precision_test_logReg.append(precision_score(Y_test, Y_test_logReg))
recall_test_logReg.append(recall_score(Y_test, Y_test_logReg))
print("Classification report for the testing set: \n", cr(Y_test, Y_test_logReg))
cm= confusion_matrix(Y_test, Y_test_logReg)
print("Confusion matrix for the testing set: \n", confusion_matrix(Y_test, Y_test_logReg))
print("Test accuracy score: " , accuracy_score(Y_test, Y_test_logReg))
print("Roc AUC for the testing set: ", roc_auc_score(Y_test, Y_test_logReg))


#-------------------------------------------------Saving the performances into external excel files ----------------------------#
# In[40]:


accuracy_test_df= pd.DataFrame(
    {'accuracy_test_rfc': accuracy_test_rfc,
     'accuracy_test_linear_svc': accuracy_test_linear_svc,
     'accuracy_test_rbf': accuracy_test_rbf,
     'accuracy_test_poly': accuracy_test_poly,
     'accuracy_test_logReg': accuracy_test_logReg

    },index= ['Raw','Best_Imp', 'Best_Imp_Sc', 'Best_All','Best_All_Sc'] )


# In[41]:


accuracy_train_df= pd.DataFrame(
    {'accuracy_train_rfc': accuracy_train_rfc,
     'accuracy_train_linear_svc': accuracy_train_linear_svc,
     'accuracy_train_rbf': accuracy_train_rbf,
     'accuracy_train_poly': accuracy_train_poly,
     'accuracy_train_logReg': accuracy_train_logReg

    },index= ['Raw','Best_Imp', 'Best_Imp_Sc', 'Best_All','Best_All_Sc'])


# In[42]:


rocAuc_test_df= pd.DataFrame(
    {'rocAuc_test_rfc': rocAuc_test_rfc,
     'rocAuc_test_linear_svc': rocAuc_test_linear_svc,
     'rocAuc_test_rbf': rocAuc_test_rbf,
     'rocAuc_test_poly': rocAuc_test_poly,
     'rocAuc_test_logReg': rocAuc_test_logReg

    },index= ['Raw','Best_Imp', 'Best_Imp_Sc', 'Best_All','Best_All_Sc'])


# In[43]:


rocAuc_train_df= pd.DataFrame(
    {'rocAuc_train_rfc': rocAuc_train_rfc,
     'rocAuc_train_linear_svc': rocAuc_train_linear_svc,
     'rocAuc_train_rbf': rocAuc_train_rbf,
     'rocAuc_train_poly': rocAuc_train_poly,
     'rocAuc_train_logReg': rocAuc_train_logReg

    },index= ['Raw','Best_Imp', 'Best_Imp_Sc', 'Best_All','Best_All_Sc'])


# In[44]:


precision_test_df= pd.DataFrame(
    {'precision_test_rfc': precision_test_rfc,
     'precision_test_linear_svc': precision_test_linear_svc,
     'precision_test_rbf': precision_test_rbf,
     'precision_test_poly': precision_test_poly,
     'precision_test_logReg': precision_test_logReg

    },index= ['Raw','Best_Imp', 'Best_Imp_Sc', 'Best_All','Best_All_Sc'] )


# In[45]:


precision_train_df= pd.DataFrame(
    {'precision_train_rfc': precision_train_rfc,
     'precision_train_linear_svc': precision_train_linear_svc,
     'precision_train_rbf': precision_train_rbf,
     'precision_train_poly': precision_train_poly,
     'precision_train_logReg': precision_train_logReg

    },index= ['Raw','Best_Imp', 'Best_Imp_Sc', 'Best_All','Best_All_Sc'] )


# In[46]:


recall_test_df= pd.DataFrame(
    {'recall_test_rfc': recall_test_rfc,
     'recall_test_linear_svc': recall_test_linear_svc,
     'recall_test_rbf': recall_test_rbf,
     'recall_test_poly': recall_test_poly,
     'recall_test_logReg': recall_test_logReg

    },index= ['Raw','Best_Imp', 'Best_Imp_Sc', 'Best_All','Best_All_Sc'] )


# In[47]:


recall_train_df= pd.DataFrame(
    {'recall_train_rfc': recall_train_rfc,
     'recall_train_linear_svc': recall_train_linear_svc,
     'recall_train_rbf': recall_train_rbf,
     'recall_train_poly': recall_train_poly,
     'recall_train_logReg': recall_train_logReg

    },index= ['Raw','Best_Imp', 'Best_Imp_Sc', 'Best_All','Best_All_Sc'] )


# In[48]: Change the path of the destination file below


writer=pd.ExcelWriter('C:\\Users\\Administrateur\\Documents\\CS\\CentraleSupelec\\2018_2019\\Projet inno\\Data\\organized data\\performance_s180_cv3_df.xlsx')


# In[49]:


accuracy_test_df.to_excel(writer, "accuracy_test_df")


# In[50]:


accuracy_train_df.to_excel(writer,'accuracy_train_df')


# In[51]:


rocAuc_test_df.to_excel(writer,
               sheet_name ='rocAuc_test_df')


# In[52]:


rocAuc_train_df.to_excel(writer,
               sheet_name ='rocAuc_train_df')


# In[53]:


precision_test_df.to_excel(writer,
               sheet_name ='precision_test_df')


# In[54]:


precision_train_df.to_excel(writer,
               sheet_name ='precision_train_df')


# In[55]:


recall_test_df.to_excel(writer,
               sheet_name ='recall_test_df')


# In[56]:


recall_train_df.to_excel(writer,
               sheet_name ='recall_train_df')


# In[57]:


writer.save()


# In[ ]:




