from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv("filtered2.csv")
y = data["Rating"]
data["Free"] = data["Free"].astype("int")
X = data.loc[:, ~data.columns.isin(['Rating', 'Minimum Android', 'Category', 'Content Rating'])]
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=0)

xTrain.to_csv("xTrain.csv", index=False)
xTest.to_csv("xTest.csv", index=False)
yTrain.to_csv("yTrain.csv", index=False)
yTest.to_csv("yTest.csv", index=False)

combined = xTrain.copy()
combined['y'] = yTrain

plt.figure(figsize=(10, 10))
sns.set(font_scale=0.8)
sns.heatmap(combined.corr(), annot=True, annot_kws={"size": 7}, fmt=".2f", cmap='coolwarm', square=True)
plt.savefig("heatmap.png", dpi = 300)

select = f_regression(xTrain, yTrain)
print("------------------F-Regression Results------------------")
np.set_printoptions(precision=3)
print("F statistics:")
print(select[0])
print("\np-values:")
print(select[1])

scaler = StandardScaler().fit(xTrain)
xTrain_norm = scaler.transform(xTrain)
xTest_norm = scaler.transform(xTest)

i = 0
explained = 0
pca = None

while explained < 0.95:
    i += 1
    pca = PCA(n_components=i)
    pca.fit(xTrain_norm)
    explained = sum(pca.explained_variance_ratio_)

print(f"A total of {i} components were kept to recover{explained * 100: 5.2f}% variance.")
print("Three most important components are:")
print(xTrain.columns.to_list())
print(pca.components_[0:3])
