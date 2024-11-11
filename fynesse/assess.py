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

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

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
