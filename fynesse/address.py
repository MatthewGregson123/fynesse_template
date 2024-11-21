# This file contains code for suporting addressing questions in the data
from .config import *
"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def k_fold_cross_validation(k, x_data, y_data, regularised, alpha, l1_wt, MSE):
  # Define size of subset
  subset_size = len(x_data) // k
  MSEs = []
  # For each combination of train and test
  for i in range(k):
    # Split data into test and train for x and y
    test_data_x = x_data[i * subset_size : (i + 1) * subset_size]
    train_data_x = np.concatenate((x_data[:i * subset_size], x_data[(i + 1) * subset_size:]))

    test_data_y = y_data[i * subset_size : (i + 1) * subset_size]
    train_data_y = np.concatenate((y_data[:i * subset_size], y_data[(i + 1) * subset_size:]))

    # Define the model using the training data
    model = sm.GLM(train_data_y, train_data_x, family=sm.families.Gaussian())
    if not regularised:
      # If we don't want a regularised model then fit normally
      fitted_model = model.fit()

      # Get the models prediction for the test data
      y_pred = fitted_model.get_prediction(test_data_x).summary_frame(alpha=0.05)
      # Get the truth for the test data
      y_true = y_data[i * subset_size : (i+1)*subset_size]
      if MSE:
        # if we want the MSE then add MSE to MSEs
        MSEs.append((np.mean((y_true - y_pred['mean']) ** 2)) ** 0.5)
      else:
        # otherwise we want correlation so add the correlation
        MSEs.append(np.corrcoef(y_true, y_pred['mean'])[0][1])
    else:
      # If we want a regularized model then fit with alpha and l1_wt
      fitted_model = model.fit_regularized(alpha=alpha, L1_wt=l1_wt)

      # Get the predictions from the model and the truth
      y_pred = fitted_model.predict(test_data_x)
      y_true = y_data[i * subset_size : (i+1)*subset_size]

      if MSE:
        # Append MSE if thats what we want
        MSEs.append((np.mean((y_true - y_pred) ** 2)) ** 0.5)
      else:
        # Otherwise append correlation coefficient
        MSEs.append(np.corrcoef(y_true, y_pred)[0][1])

  MSEs = np.array(MSEs)
  # Return mean and Standard Deviation of result
  return np.mean(MSEs), np.std(MSEs)

  def plot_alpha_MSE(x_data,y_data, l1_wt, min_alpha, max_alpha, steps, title):
  alpha_correlations = []
  alpha_MSE = []
  # splits our range of alphas into steps
  for i in range(steps):
    # Get the average RMSE over 5 folds with the given alpha and l1_wt
    mean, _ = k_fold_cross_validation(5, x_data, y_data, True, min_alpha + i * (max_alpha - min_alpha) / steps, l1_wt, MSE=True)
    alpha_MSE.append(mean)
    # Get the average correlation over 5 folds with given alpha and l1_wt
    mean, _ = k_fold_cross_validation(5, x_data, y_data, True, min_alpha + i * (max_alpha - min_alpha) / steps, l1_wt, MSE=False)
    alpha_correlations.append(mean)

  # Create subplot
  fig, axs = plt.subplots(1, 2)

  xs = np.linspace(min_alpha,max_alpha,steps)
  
  # Plot the MSE vs Alpha
  axs[0].plot(xs, alpha_MSE)
  axs[0].set_ylabel("Mean Squared Error")
  axs[0].set_xlabel("Alpha")
  axs[0].set_title(title + " MSE")
  axs[0].set_ylim(min(alpha_MSE) * 0.9, max(alpha_MSE) * 1.1)

  # Plot the Correlation vs Alpha
  axs[1].plot(xs, alpha_correlations)
  axs[1].set_ylabel("Correlation")
  axs[1].set_xlabel("Alpha")
  axs[1].set_title(title + " Correlation")
  axs[1].set_ylim(min(alpha_correlations) * 0.9, max(alpha_correlations) * 1.1)

  # Show the graph
  plt.tight_layout()
  plt.show()

  def predict_profile(NS_SEC, truth, norm_age_df, x_data, alpha, l1_wt):
  preds = []
  # Repeat for all ages
  for i in range(0, 100):
    # Get the y_data
    y_data = norm_age_df[i].to_numpy()

    # Create and fit a regulariazed model with alpha and l1_wt 
    model = sm.GLM(y_data, x_data, family=sm.families.Gaussian())
    fitted_model = model.fit_regularized(alpha=alpha, L1_wt=l1_wt)

    # Get the prodictions given the NS_SEC data
    pred = fitted_model.predict(NS_SEC)
    preds.append(pred)
  
  # Plot the truth against the prediction
  plt.plot(np.arange(0, 100), list(truth), label = "Truth")
  plt.plot(np.arange(0, 100), preds, label = "Prediction")
  plt.xlabel("age")
  plt.ylabel("share of population")

  # Show the graph
  plt.legend()
  plt.show()
  
  
