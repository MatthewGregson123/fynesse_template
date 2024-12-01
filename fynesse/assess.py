from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""
import matplotlib.pyplot as plt
import osmnx as ox
import scipy.stats as stats
import numpy as np
import math
import seaborn as sns
import random
import pandas as pd

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

def k_means(pois_df, k):
  # Get length of dataset
  length_dataset = len(pois_df)

  # If we want more clusters then there are data items it's not possible
  if length_dataset < k:
    print("Not possible")

  # Initialise clusters
  clusters = {}
  chosen = []
  # Choose random nodes to act as start points
  while len(chosen) < k:
    choice = random.randint(0, length_dataset - 1)
    # Make sure its not already chosen
    if choice not in chosen:
      chosen.append(choice)
      # Set the dimension vector of the chosen start point to the chosen nodes vector
      clusters[len(chosen) - 1] = pois_df.iloc[choice].to_dict()

  # Initial assignemnt is empty
  cluster_assignment = {}
  changing = True
  # Keep looping until we see no change
  while changing:
    # Create a copy of the cluster assignment to check if change happened
    old_cluster_assignment = cluster_assignment.copy()

    # Assign nodes to nearest cluster centre
    cluster_assignment = {}
    # Keep track of names
    cluster_assignment_names = {}
    for i in range(length_dataset):
      # Current node we're examining
      node =  pois_df.iloc[i].to_dict()
      node_name = pois_df.index[i]

      # Initialise values for checking for closest cluster
      min_distance = math.inf
      min_index = 0
      # Loop through all clusters
      for j in range(k):
        curr_distance = 0
        for key in clusters[j]:
          # calculate distance to current cluster we're examining
          curr_distance += (clusters[j][key] - node[key]) ** 2
        # if the distance is less than the min_distance we update values
        if curr_distance < min_distance:
          min_distance = curr_distance
          min_index = j

      # Add the node to the min_index cluster
      if min_index in cluster_assignment:
        cluster_assignment[min_index].append(node)
      else:
        cluster_assignment[min_index] = [node]

      # Add the name to the min_index cluster too
      if min_index in cluster_assignment_names:
        cluster_assignment_names[min_index].append(node_name)
      else:
        cluster_assignment_names[min_index] = [node_name]

    # Adjust cluster centres
    for cluster in cluster_assignment:
      n_nodes = len(cluster_assignment[cluster])
      # Set all values to 0
      for tag in clusters[cluster]:
        clusters[cluster][tag] = 0

      # For each node
      for node in cluster_assignment[cluster]:
        # For each tag in each node
        for tag in clusters[cluster]:
          # Add it to the clusters value for tag divided by n_nodes for the mean
          clusters[cluster][tag] +=node[tag]/n_nodes

    # If nothing has changed then
    if old_cluster_assignment == cluster_assignment:
      changing = False

  return cluster_assignment_names

def compare_clusters_to_expected(expected, trials, data, k):
  # keeps track of how many time the clusters meet what we expected
  trials_passed = []
  for _ in range(trials):
    # Compute the clusters using the k_means algorithm
    computed = k_means(data, k)

    identical = True
    # loop through all the clusters we expect to see
    for cluster in expected:
      # initialise values before search
      node = expected[cluster]
      node_found = False

      # iterate through clutsers that we computerd
      for key in computed:
        # create a copy of current cluster
        temp = list(computed[key])
        # keep track of number of items we popped
        count = 0
        # loop until we've popped all values in temp
        while len(temp) > 0:
          # check if the current node in the computed cluster is in the expected
          if temp[0] in node:
            # if it is increment the count and pop it
            count += 1
            temp.pop(0)
          else:
            # if its not then break as they aren't the same cluster
            break
        # if the count is the length of the node it means that all nodes in cluster 
        # have been matched as we can't have duplicate nodes in a cluster.
        # if the len of temp is 0 then we know all nodes in the computed cluster have
        # been matched. Therefore they are identical clusters
        if count == len(node) and len(temp) == 0:
          # Say we've found that node / cluster 
          node_found = True
      # if we haven't found the node / cluster than the cluster groupings aren't 
      # the same so set identical to false
      if not node_found:
        identical = False

    # update whether the computed cluster is the same as what we expected based on identical
    if identical:
      trials_passed.append(1)
    else:
      trials_passed.append(0)
  # print mean trials passed and standard deviation
  print("Meeted expectation: ", np.mean(trials_passed), " +/- ", np.std(trials_passed))

def compute_distance_matrix(data):
  # Initialise values
  distance_matrix = []
  examined = ["node"]
  index_to_distance_matrix_entry = {}
  columns = []
  # iterate through the rows in the data
  for index1, _ in data.iterrows():
    # Add an entry to the distance_matrix for the index if its not in
    # Also add it to the index_to_distance_matrix_entry
    if index1 not in index_to_distance_matrix_entry:
      distance_matrix.append({"node": index1, index1: 0})
      index_to_distance_matrix_entry[index1] = len(distance_matrix) - 1

    # Get the data for the node we're examining as a node
    node1 = data.loc[index1].to_dict()
    # Append the index to the examined
    examined.append(index1)

    # Loop through all other nodes
    for index2, _ in data.iterrows():
      # Matrix is symetrical along diagonal so don't need to check indexes we've already examined
      if index2 not in examined:
        if index1 != index2:
          # Get the data for the node we're examining
          node2 = data.loc[index2].to_dict()
          # Set the intital distance to 0
          distance = 0
          # Calculate the distance of the 2 nodes
          for key in node1:
            distance += (node1[key] - node2[key]) ** 2
          distance = distance ** 0.5

          # make sure both nodes are in the distance matrix
          if index2 not in index_to_distance_matrix_entry:
            distance_matrix.append({"node": index2, index2: 0})
            index_to_distance_matrix_entry[index2] = len(distance_matrix) - 1

          # update distance_matrix values
          distance_matrix[index_to_distance_matrix_entry[index1]][index2] = distance
          distance_matrix[index_to_distance_matrix_entry[index2]][index1] = distance

  # Create a dataframe of the distance matrix
  distance_matrix_df = pd.DataFrame(distance_matrix, columns=examined)

  # Set the node as the index
  distance_matrix_df.set_index("node", inplace=True)
  return distance_matrix_df

def plot_heat_map(data, xlabel, ylabel, title):
  # set the size of the figure
  plt.figure(figsize=(12,10))
  # plot the heatmap with the colour map virdis, annotations on the cell, 2 d.p, and 0.5 linewidth
  sns.heatmap(data, cmap='viridis', annot=True, fmt=".2f", linewidths=0.5)

  # set title and axis labels
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  # show the plot
  plt.show()


def plot_map(area,edges, buildings_with_addresses_pois, buildings_without_addresses_pois, north, south, west, east):
  fig, ax = plt.subplots()

  # Plot the footprint
  area.plot(ax=ax, facecolor="white")

  # Plot street edges
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

  ax.set_xlim([west, east])
  ax.set_ylim([south, north])
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")

  # Plot tourist places
  buildings_with_addresses_pois.plot(ax=ax, color="blue", alpha=1, markersize=10)
  buildings_without_addresses_pois.plot(ax=ax, color="purple", alpha=1, markersize=5)
  plt.tight_layout()

def plot_df_columns(data, column_x, column_y):
  x = []
  y = []
  for i in range(len(data)):
    row = data.iloc[i].to_dict()
    if column_x in row and column_y in row:
      x.append(row[column_x])
      y.append(row[column_y])

  fig, ax = plt.subplots(figsize=(7, 7))

  plt.scatter(x, y)
  plt.xlabel(column_x)
  plt.ylabel(column_y)

  b, a = np.polyfit(x, y, deg=1)
  xs = np.linspace(0, math.ceil(max(x)), 100)
  ys = a + b * xs
  ax.plot(xs, ys, color="k", lw=2.5)

  plt.show()

  corr, _ = stats.pearsonr (x, y)
  return corr

def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
