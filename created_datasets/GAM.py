import sys
from pygam import LinearGAM, s, f, te
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATA = "/Users/maddiedailey/Desktop/wr.csv"


def data_processing(data, output):
    """
    Generate train-test-split data using output as desired predicted outcome

    data: dataset to process
    output: desired outcome feature
    """
    y = data[output]
    X = data.drop(output, axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
    return X_train, X_test, y_train, y_test

def main():

    # read in dataset
    data = pd.read_csv(DATA)

    # remove categorical data
    data = data.drop("Name", axis = 1)
    data = data.drop("Week", axis = 1)
    #data = data.drop("total_receiving_yards_allowed", axis=1)

    # train two models for each of our performance metrics
    td_x_train, td_x_test, td_y_train, td_y_test = data_processing(data, "Receiving.TDs.x")
    ry_x_train, ry_x_test, ry_y_train, ry_y_test = data_processing(data, "Receiving.Yards.x")

    print("Fitting Model...")

    td_gam = LinearGAM().fit(td_x_train, td_y_train)
    #ry_gam = LinearGAM().fit(ry_x_train, ry_y_train)

    # Perform cross-validation for each df value
    # lam_values = [0.01, 0.1, 0.5, 0.6, 0.8]
    # cv_scores = []
    # for lam in lam_values:
    #     gam = LinearGAM(s(0) + te(1,3) + te(2,5) + s(4) + s(5) + s(6) + te(7,8) + s(9) + s(10) + s(11)).fit(ry_x_train, ry_y_train)
    #     y_pred = gam.predict(ry_x_test)
    #     mse = mean_squared_error(ry_y_test, y_pred)
    #     cv_scores.append(mse)

    # # Find the optimal df value with the lowest MSE
    # optimal_lam = lam_values[np.argmin(cv_scores)]
    # print("Optimal Lambda:", optimal_lam)

    ry_gam = LinearGAM(s(0) + te(1,3) + te(2,5) + s(4) + s(5) + s(6) + te(7,8) + s(9) + 
                       s(10) + s(11) + s(12)).fit(ry_x_train, ry_y_train)

    print("TD Model: ")
    print(td_gam.summary())
    print("Recieving Yards Model: ")
    print(ry_gam.summary())

    td_y_pred = td_gam.predict(td_x_test)
    ry_y_pred = ry_gam.predict(ry_x_test)

    td_mse = mean_squared_error(td_y_test, td_y_pred)
    ry_mse = mean_squared_error(ry_y_test, ry_y_pred)

    print("TD Model Performance: (MSE)")
    print(td_mse)
    print("Recieving Yards Model Performance: (MSE)")
    print(ry_mse)



if __name__ == '__main__':
    main()

