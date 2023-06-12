import sys
from pygam import LinearGAM, s, f, te
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


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

    # train two models for each of our performance metrics
    td_x_train, td_x_test, td_y_train, td_y_test = data_processing(data, "TD")
    ry_x_train, ry_x_test, ry_y_train, ry_y_test = data_processing(data, "Receiving.Yards")

    td_gam = LinearGAM().fit(td_x_train, td_y_train)
    ry_gam = LinearGAM().fit(ry_x_train, ry_y_train)

    td_y_pred = td_gam.predict(td_x_test)
    ry_y_pred = ry_gam.predict(ry_x_test)

    print(td_gam.summary())
    print(ry_gam.summary())

    td_metrics = confusion_matrix(td_y_test, td_y_pred)
    ry_metrics = confusion_matrix(ry_y_test, ry_y_pred)

    print(td_metrics)
    print(ry_metrics)



if __name__ == '__main__':
    main()

