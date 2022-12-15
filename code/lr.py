import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def main():
    data = pd.read_csv("filtered2.csv")
    y = data["Rating"].to_numpy()
    data = data.drop(["Category", "Free", "Minimum Android", "Content Rating"], axis=1)
    Xs = data.drop(["Rating"], axis=1).to_numpy()

    # split dataset
    xTrain, xTest, yTrain, yTest = train_test_split(Xs, y, test_size=0.2)

    scaler=StandardScaler()
    xTrain=scaler.fit_transform(xTrain)
    xTest=scaler.transform(xTest)

    model=LinearRegression()
    model.fit(xTrain,yTrain)
    predicted=model.predict(xTest)
    print(mean_squared_error(yTest,predicted,squared=True))




if __name__ == "__main__":
    main()