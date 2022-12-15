from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


xTrain = pd.read_csv("xTrain_cat_std.csv")
yTrain = pd.read_csv("yTrain_cat.csv")
xTest = pd.read_csv("xTest_cat_std.csv")
yTest = pd.read_csv("yTest_cat.csv")

n_estimators = [90, 100, 120, 150] # number of trees in the random forest
max_features = ['auto', 'sqrt'] # number of features in consideration at every split
max_depth = [140, 150, 160] # maximum number of levels allowed in each decision tree
min_samples_split = [1, 2, 3] # minimum sample number to split a node
min_samples_leaf = [4, 5, 6] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points

random_grid = { 'n_estimators': n_estimators,

                'max_features': max_features,

                'max_depth': max_depth,

                'min_samples_split': min_samples_split,

                'min_samples_leaf': min_samples_leaf,

                'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,
               n_iter = 10, cv = 5, verbose=2, random_state=0, n_jobs = -1)
rf_random.fit(xTrain, yTrain)
print ('Best Parameters: ', rf_random.best_params_, ' \n')

rf = RandomForestRegressor(n_estimators=170, min_samples_split=2, min_samples_leaf=5, max_features='sqrt', max_depth=200, bootstrap=True)
rf.fit(xTrain, yTrain)

# code from https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
import time
import numpy as np

feature_names = xTrain.columns

start_time = time.time()
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

import pandas as pd

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig('importance.png', dpi=300)

yHat = rf.predict(xTest)
print(mean_squared_error(yHat, yTest, squared=False))
print(r2_score(yHat, yTest))