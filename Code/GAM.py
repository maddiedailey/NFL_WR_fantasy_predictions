import sys
from pygam import LinearGAM, PoissonGAM, s, f, te
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv

# Filepath to dataset
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
    # data = data.drop("total_receiving_yards_allowed", axis=1)

    # train two models for each of our performance metrics
    td_x_train, td_x_test, td_y_train, td_y_test = data_processing(data, "Receiving.TDs.x")
    ry_x_train, ry_x_test, ry_y_train, ry_y_test = data_processing(data, "Receiving.Yards.x")

    print("Fitting Model...")

    td_gam = PoissonGAM().fit(td_x_train, td_y_train)
    #td_gam = LinearGAM().fit(td_x_train, td_y_train)

    ry_gam = LinearGAM().fit(ry_x_train, ry_y_train)

    # OPTIMIZATION #####

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

    #ry_gam = LinearGAM(s(0) + te(1,3) + te(2,5) + s(4) + s(5) + s(6) + te(7,8) + s(9) + 
    #                   s(10) + s(11) + s(12)).fit(ry_x_train, ry_y_train)

    ######################

    print("\nTD Model: \n")
    print(td_gam.summary())
    print("\nRecieving Yards Model: \n")
    print(ry_gam.summary())

    td_y_pred = td_gam.predict(td_x_test)
    ry_y_pred = ry_gam.predict(ry_x_test)

    td_mse = mean_squared_error(td_y_test, td_y_pred)
    ry_mse = mean_squared_error(ry_y_test, ry_y_pred)

    print("\nTD Model Performance: (MSE)\n")
    print(td_mse)
    print("\nRecieving Yards Model Performance: (MSE)\n")
    print(ry_mse)


    corr_matrix = np.corrcoef(td_y_test, td_y_pred)
    corr = corr_matrix[0,1]
    td_R_sq = corr**2
    
    print("TD R2: \n")
    print(td_R_sq)

    corr_matrix = np.corrcoef(ry_y_test, ry_y_pred)
    corr = corr_matrix[0,1]
    ry_R_sq = corr**2
    
    print("RY R2: \n")
    print(ry_R_sq)

    # PLOTS ########
    # Create TD plots
    plt.scatter(td_y_test, td_y_pred, color='blue', alpha=1, label='TD Data Points')

    # Add a red line overlaying the plot
    plt.plot([min(td_y_test), max(td_y_test)], [min(td_y_test), max(td_y_test)], color='red', linestyle='-', linewidth=2, label='Ideal')
    
    # Set labels and title
    plt.xlabel('True TD Values')
    plt.ylabel('Predicted TD Values')
    plt.title('Scatter Plot: True vs. Predicted')

    # Display the plot
    plt.show()

    # plt.boxplot(box_mat)

    # plt.xlabel('True TD Values')
    # plt.ylabel('Predicted TD Values')
    # plt.title('Scatter Plot: True vs. Predicted')

    # # Display the plot
    # plt.show()

    # Create RY plots
    # plt.scatter(ry_y_test, ry_y_pred, color='blue', alpha=0.5, label='RY Data Points')

    # # Add a red line overlaying the plot
    # plt.plot([min(ry_y_test), max(ry_y_test)], [min(ry_y_test), max(ry_y_test)], color='red', linestyle='-', linewidth=2, label='Ideal')
    
    # # Set labels and title
    # plt.xlabel('True RY Values')
    # plt.ylabel('Predicted RY Values')
    # plt.title('Scatter Plot: True vs. Predicted')

    # # Display the plot
    # plt.show()
    ###################



if __name__ == '__main__':
    main()

