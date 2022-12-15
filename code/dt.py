import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def main():
    data=pd.read_csv("filtered2.csv")
    y=data["Rating"].to_numpy()
    data=data.drop(["Category","Free","Minimum Android","Content Rating"],axis=1)
    Xs=data.drop(["Rating"],axis=1).to_numpy()

    # split dataset
    xTrain,xTest,yTrain,yTest=train_test_split(Xs,y,test_size=0.2)

    # find best depth
    max_depth=41
    depths=list(range(3,max_depth))
    mse=[]
    for depth in depths:
        tree=DecisionTreeRegressor(max_depth=depth)
        tree.fit(xTrain,yTrain)
        predicted=tree.predict(xTest)
        mse.append(mean_squared_error(yTest,predicted,squared=False))
    plt.plot(depths,mse)
    plt.xlabel("depth")
    plt.ylabel("MSE")
    plt.savefig("dt-depth.png")
    plt.show()

    # find best min samples
    max_samples=500000
    samples=list(range(100,max_samples+1,2500))
    mse=[]
    optimal=0
    error=20
    for sample in samples:
        tree=DecisionTreeRegressor(max_depth=10,min_samples_leaf=sample)
        tree.fit(xTrain,yTrain)
        predicted=tree.predict(xTest)
        e=mean_squared_error(yTest,predicted,squared=False)
        mse.append(e)
        if(e<error):
            error=e
            optimal=sample
        plt.plot(samples,mse)
        plt.xlabel("minimum leaf samples")
        plt.ylabel("MSE")
        plt.savefig("dt-leaf.png")
        print(error)
        print(optimal)
        plt.show()

    model=DecisionTreeRegressor(max_depth=10,min_samples_leaf=100)
    model.fit(xTrain,yTrain)
    predicted=model.predict(xTest)
    print(mean_squared_error(yTest,predicted,squared=True))







if __name__ == "__main__":
    main()