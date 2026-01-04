import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler,StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

def evaluate_models(X, y, title):
    # Ensure X is a 2D array as required by sklearn
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    # # Sort X and y based on X values
    # sorted_indices = np.argsort(X, axis=0).ravel()
    # X = X[sorted_indices]
    # y = y[sorted_indices]
    
    # Normalize X and y between 0 and 1
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y)


    # Split the data into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare models
    models = {
        "Linear Regression": LinearRegression().fit(X_train, y_train),
        "Polynomial Regression (degree=2)": None,
        "Decision Tree (max_depth=7)": DecisionTreeRegressor(max_depth=7).fit(X_train, y_train),
        "K-Nearest Neighbors (k=12)": KNeighborsRegressor(n_neighbors=12).fit(X_train, y_train),
        "Support Vector Regression (RBF)": SVR(kernel='rbf').fit(X_train, y_train),
        "Gaussian Process Regression": None
    }

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    models["Polynomial Regression (degree=2)"] = poly_model

    # Gaussian Process Regression
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
    gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr_model.fit(X_train, y_train)
    models["Gaussian Process Regression"] = gpr_model

    # Evaluate and plot results
    plt.figure(figsize=(12, 12))
    for i, (name, model) in enumerate(models.items()):
        if name == "Polynomial Regression (degree=2)":
            y_train_pred = model.predict(X_train_poly)
            y_test_pred = model.predict(X_test_poly)
        elif name == "Gaussian Process Regression":
            y_train_pred, sigma_train = model.predict(X_train, return_std=True)
            y_test_pred, sigma_test = model.predict(X_test, return_std=True)
        else:
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

        r2_train = r2_score(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)

        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)

        # Print evaluation metrics
        print(f"{name}")
        print(f"  Train R2 Score: {r2_train:.4f}, Train MAE: {mae_train:.4f}")
        print(f"  Test R2 Score: {r2_test:.4f}, Test MAE: {mae_test:.4f}\n")

        # Plot results
        plt.subplot(3, 2, i + 1)
        plt.scatter(X_train, y_train, color='blue', label='Train Data')
        plt.scatter(X_test, y_test, color='green', label='Test Data')
        plt.plot(np.sort(X_train, axis=0), np.sort(y_train_pred, axis=0), color='red', label='Train Prediction')
        plt.plot(np.sort(X_test, axis=0), np.sort(y_test_pred, axis=0), color='orange', label='Test Prediction')
        plt.title(name)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()

    plt.tight_layout()
    plt.savefig("/sise/home/lionb/figures/logistic_regression_{}.png".format(title))

# corr_df = pd.read_csv("/sise/home/lionb/cell_generator/figures/membrane_correlation.csv")
# corr_df = corr_df.dropna(how='any')

# title = "Plasma-Membrane"
# x_col = '../mg_model_membrane_13_05_24_1.5'
# y_col = '../unet_model_22_05_22_membrane_128'

# corr_df = corr_df[corr_df[y_col]>0.2]

# x = corr_df[x_col].values.reshape([-1,1])
# y = corr_df[y_col].values.reshape([-1,1])

# Perturbations data
corr_df = pd.read_csv("/groups/assafza_group/assafza/full_cells_fovs_perturbation/train_test_list/unet_predictions/metadata_with_efficacy_scores_and_unet_scores.csv")
drug_label = 'All'

x_col = '../mg_model_dna_13_05_24_1.5b'
y_col = '../unet_model_22_05_22_dna_128'

# corr_df = corr_df[(corr_df['drug_label'] == drug_label) & (corr_df[y_col]>0.0)]
x = corr_df[x_col].values.reshape([-1,1])
y = corr_df[y_col].values.reshape([-1,1])
title = "DNA-perturbations-{}".format(drug_label)

evaluate_models(x,y,title)