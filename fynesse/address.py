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
from . import access
from . import assess
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random
import pandas as pd
import warnings

def k_fold_cross_validation(k, x_data, y_data, regularised, alpha, l1_wt, MSE):
  # Define size of subset
  subset_size = len(x_data) // k
  preds = []
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

      # Append prediction to the list of predictions
      preds.extend(y_pred['mean'])
    else:
      # If we want a regularized model then fit with alpha and l1_wt
      fitted_model = model.fit_regularized(alpha=alpha, L1_wt=l1_wt)

      # Get the predictions from the model and the truth
      y_pred = fitted_model.predict(test_data_x)
      
      # Append prediction to the list of predictions
      preds.extend(y_pred)

  # Return correlation or MSE
  preds = np.array(preds)
  if MSE:
    return np.mean((y_data[:len(preds)] - preds) ** 2)
  else:
    return np.corrcoef(y_data[:len(preds)], preds)[0][1]

def plot_alpha_MSE(x_data,y_data, l1_wt, min_alpha, max_alpha, steps, title):
  alpha_correlations = []
  alpha_MSE = []
  # splits our range of alphas into steps
  for i in range(steps):
    # Get the average RMSE over 5 folds with the given alpha and l1_wt
    mean = k_fold_cross_validation(5, x_data, y_data, True, min_alpha + i * (max_alpha - min_alpha) / steps, l1_wt, MSE=True)
    alpha_MSE.append(mean)
    # Get the average correlation over 5 folds with given alpha and l1_wt
    mean = k_fold_cross_validation(5, x_data, y_data, True, min_alpha + i * (max_alpha - min_alpha) / steps, l1_wt, MSE=False)
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

def get_population_density(lat, long, cursor):
  north, south, west, east = access.create_bounding_box(lat, long, 1)
  cursor.execute(f"SELECT pdd.density, pdd.geography, gcd.Shape_Area, gcd.lat, gcd.`LONG`  FROM population_density_data AS pdd INNER JOIN geo_coords_data AS gcd ON pdd.geography = gcd.OA21CD WHERE gcd.LAT <{north} AND gcd.LAT >{south} AND gcd.`LONG` < {east} AND gcd.`LONG` > {west}")
  density_df = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])

  density = density_df[["density"]].to_numpy()
  area = density_df[["Shape_Area"]].to_numpy()
  density_data = (density * area) / np.sum(area)

  total_density = np.sum(density_data)

  return total_density

def create_train_data_L15_pop_density(locations_dict, username, password, url):
  x_train = []
  x_train_density = []
  y_train = []
  y_train_density = []

  conn =  access.create_connection(username, password, url, database='ads_2024')
  cursor = conn.cursor()


  query = f"SELECT LAT, `LONG` FROM geo_coords_data"
  cursor.execute(query)

  c = 0
  target = 100

  indexes = []

  row = cursor.fetchall()
  i = 0
  j = 0

  must_examine = [(51.5189, -0.09083), (51.519, -0.09649),(51.5138, -0.09959), (51.5171, -0.09361), (51.5208, -0.09625)]
  while c < target or i < len(locations_dict) or j < len(must_examine):
    print(c, end=', ')
    if c < target:
      index = random.randint(0, len(row)-1)
      while index in indexes:
        index = random.randint(0, len(row)-1)
      indexes.append(index)
      lat, long = row[index]
    elif i < len(locations_dict):
      lat, long = locations_dict[list(locations_dict.keys())[i]]
      i += 1
    else:
      lat, long = must_examine[j]
      j += 1
    poi_counts_df =  assess.get_pois_counts_from_sql({"location": (lat, long)}, cursor)
    c += 1
    norm_poi_counts_df = poi_counts_df.div(poi_counts_df.sum(axis=1), axis=0)


    nsec_df =  assess.get_nsec_df_for_locations({"location": (lat, long)}, cursor)
    nsec_df_noL15 = nsec_df.drop('L15',axis=1)
    norm_nsec_df = nsec_df.div(nsec_df.sum(axis=1), axis=0)
    norm_nsec_df_noL15 = nsec_df_noL15.div(nsec_df_noL15.sum(axis=1), axis=0)

    norm_nsec_poi_counts_df = pd.concat([norm_nsec_df, norm_poi_counts_df], axis=1)
    norm_nsec_poi_counts_df_noL15 = pd.concat([norm_nsec_df_noL15, norm_poi_counts_df], axis=1)

    train_row = norm_nsec_poi_counts_df_noL15.to_numpy()
    train_row = train_row[0]
    train_row = np.nan_to_num(train_row, nan=0)

    x_train.append(train_row)
    y_train.append(norm_nsec_poi_counts_df['L15'])

    density = get_population_density(lat, long, cursor)
    y_train_density.append(density)

    # train_row = norm_nsec_poi_counts_df.to_numpy()
    train_row = pd.concat([nsec_df, poi_counts_df], axis=1).to_numpy()
    train_row = train_row[0]
    train_row = np.nan_to_num(train_row, nan=0)

    x_train_density.append(train_row)


  x_train = np.array(x_train)
  x_train_density = np.array(x_train_density)

  y_train = np.array(y_train)
  y_train_density = np.array(y_train_density)

  return x_train, y_train, x_train_density, y_train_density

def estimate_students(latitude: float, longitude: float) -> float:
    """
    Args:
    latitude (float): The latitude coordinate.
    longitude (float): The longitude coordinate.

    Returns:
    float: Estimated share of students in that area (value between 0 and 1).
    """

    tags = {
    "amenity": ["university"],
    "historic": True,
    "leisure": True,
    "tourism": True,
    "cuisine": ["coffee_shop"],
    "office": True,
    "parking": True,
    }
    with open("credentials.yaml") as file:
      credentials = yaml.safe_load(file)
    username = credentials["username"]
    password = credentials["password"]
    url = credentials["url"]
    port = credentials["port"]

    conn = fynesse.access.create_connection(username, password, url, database='ads_2024')
    cursor = conn.cursor()

    poi_counts_df = fynesse.assess.get_pois_counts_from_sql({"location": (latitude, longitude)}, cursor)
    norm_poi_counts_df = poi_counts_df.div(poi_counts_df.sum(axis=1), axis=0)
    norm_poi_counts_df = norm_poi_counts_df.fillna(0)

    nsec_df = fynesse.assess.get_nsec_df_for_locations({"location": (latitude, longitude)}, cursor)
    nsec_df = nsec_df.drop('L15', axis=1)
    norm_nsec_df = nsec_df.div(nsec_df.sum(axis=1), axis=0)
    norm_nsec_df = norm_nsec_df.fillna(0)


    norm_nsec_poi_counts_df = pd.concat([norm_nsec_df, norm_poi_counts_df], axis=1)
    x_test = norm_nsec_poi_counts_df.to_numpy()

    model = sm.GLM(y_train, x_train, family=sm.families.Gaussian())
    fitted_model = model.fit()

    y_pred = fitted_model.predict(x_test)

    cursor.close()
    conn.close()


    return y_pred[0]


def estimate_population_density(latitude: float, longitude: float) -> float:
    """
    Args:
    latitude (float): The latitude coordinate.
    longitude (float): The longitude coordinate.

    Returns:
    float: Estimated value, percentage, probability, etc
    """

    tags = {
    "amenity": ["university"],
    "historic": True,
    "leisure": True,
    "tourism": True,
    "cuisine": ["coffee_shop"],
    "office": True,
    "parking": True,
    }
    with open("credentials.yaml") as file:
      credentials = yaml.safe_load(file)
    username = credentials["username"]
    password = credentials["password"]
    url = credentials["url"]
    port = credentials["port"]

    conn = fynesse.access.create_connection(username, password, url, database='ads_2024')
    cursor = conn.cursor()

    poi_counts_df = fynesse.assess.get_pois_counts_from_sql({"location": (latitude, longitude)}, cursor)

    nsec_df = fynesse.assess.get_nsec_df_for_locations({"location": (latitude, longitude)}, cursor)

    nsec_poi_counts_df = pd.concat([nsec_df, poi_counts_df], axis=1)
    nsec_poi_counts_df = nsec_poi_counts_df.fillna(0)

    x_test = nsec_poi_counts_df.to_numpy()

    model = sm.GLM(y_train_density, x_train_density, family=sm.families.Gaussian())
    fitted_model = model.fit()

    y_pred = fitted_model.predict(x_test)

    cursor.close()
    conn.close()

    return y_pred[0]

def evaluate_L15_density_model(model_name, username, password, url, locations_dict):
  indexes = []
  errors = []

  ys = []
  y_preds = []

  conn =  access.create_connection(username, password, url, database='ads_2024')
  cursor = conn.cursor()

  query = f"SELECT LAT, `LONG` FROM geo_coords_data"
  cursor.execute(query)
  rows = cursor.fetchall()

  c = 0
  target = 100
  i = 0
  while c < target or i < len(locations_dict):

    if c < target:
      index = random.randint(0, len(x_train)-1)
      while index in indexes:
        index = random.randint(0, len(x_train)-1)
      indexes.append(index)
      lat, long = rows[index]
      c+=1
    else:
      lat, long = locations_dict[list(locations_dict.keys())[i]]
      i+=1


    if model_name == "students":
      nsec_df =  assess.get_nsec_df_for_locations({"location": (lat, long)}, cursor)
      norm_nsec_df = nsec_df.div(nsec_df.sum(axis=1), axis=0)
      y = norm_nsec_df['L15'][0]
      y_pred = estimate_students(lat, long)
    elif model_name == "density":
      y = get_population_density(lat, long, cursor)
      y_pred = estimate_population_density(lat, long)
    else:
      raise ValueError("Invalid model name")

    ys.append(y)
    y_preds.append(y_pred)

    if (math.isnan(y-y_pred)):
      print(lat, long)


  cursor.close()
  conn.close()

  ys = np.array(ys)
  y_preds = np.array(y_preds)

  errors = ys - y_preds

  print("Max Error: ", np.max(errors))
  print("Min Error: ", np.min(errors))
  errors_copy = np.array(errors)
  errors_copy = (errors_copy ** 2) ** 0.5
  print("Average Error: ", np.mean(errors_copy))
  print("Correlation: ", np.corrcoef(ys, y_preds)[0, 1])
  fig, axs = plt.subplots(1, 3, figsize=(15, 5))
  axs[0].hist(errors,bins=10)
  axs[0].set_xlabel('Error')
  axs[0].set_ylabel('Frequency')
  axs[0].set_title('Histogram of Errors')

  axs[1].scatter(y_preds, ys)
  axs[1].set_xlabel('Actual')
  axs[1].set_ylabel('Predicted')
  axs[1].set_title('Actual vs Predicted')

  max_val = max(max(ys),max(y_preds))
  b, a = np.polyfit(ys, y_preds, deg=1)
  xs = np.linspace(0, max_val, 20)
  ys = a + b * xs
  axs[1].plot(xs, ys, color='red', linestyle='dotted', label="line of best fit")
  axs[1].set_ylim(0, max_val)
  axs[1].set_xlim(0, max_val)
  axs[1].plot(np.linspace(0, max(max(ys),max(y_preds)), 20), np.linspace(0, max(max(ys),max(y_preds)), 20), color='red', label="y=x")

  axs[1].legend()
  plt.show()

  if model_name == "students":
    model = sm.GLM(y_train, x_train, family=sm.families.Gaussian())
  else:
    model = sm.GLM(y_train_density, x_train_density, family=sm.families.Gaussian())
  fitted_model = model.fit()
  params = list(fitted_model.params)
  axs[2].plot(np.linspace(1, len(params), len(params)), params)
  axs[2].set_xlabel('Parameter')
  axs[2].set_ylabel('Value')
  axs[2].set_title('Parameter vs Value')

  plt.show()
  
