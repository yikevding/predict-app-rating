from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


xTrain = pd.read_csv("xTrain.csv")
yTrain = pd.read_csv("yTrain.csv")
n_estimators = [5,20,50,100] # number of trees in the random forest
max_features = ['auto', 'sqrt'] # number of features in consideration at every split
max_depth = list(range(10, 200, 10)) # maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10, 15] # minimum sample number to split a node
min_samples_leaf = [1, 3,5, 7] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points

random_grid = { 'n_estimators': n_estimators,

                'max_features': max_features,

                'max_depth': max_depth,

                'min_samples_split': min_samples_split,

                'min_samples_leaf': min_samples_leaf,

                'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,
               n_iter = 100, cv = 5, verbose=2, random_state=0, n_jobs = -1)
rf_random.fit(xTrain, yTrain)
print ('Best Parameters: ', rf_random.best_params_, ' \n')