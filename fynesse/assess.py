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
from sklearn import decomposition
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

def get_nsec_df_for_locations(locations_dict, cursor):
    nsec_locations = []
    for location in locations_dict:
      latitude, longitude = locations_dict[location]
      north, south, west, east = access.create_bounding_box(latitude, longitude, 1)
      query = f"SELECT * FROM nsec_data WHERE geography IN (SELECT OA21CD FROM geo_coords_data WHERE LAT <{north} AND LAT >{south} AND `LONG` < {east} AND `LONG` > {west})"
      cursor.execute(query)
      nsec_df = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])
      nsec_data = nsec_df[["L1_L2_L3", "L4_L5_L6", "L7", "L8_L9", "L10_L11", "L12", "L13", "L14", "L15"]]
      nsec_data = nsec_data.sum()
      nsec_locations.append(nsec_data.to_dict())
    nsec_locations_df = pd.DataFrame(nsec_locations, index=locations_dict.keys())
    return nsec_locations_df

def get_pois_counts_from_sql(locations_dict, cursor):
  pois_locations = []
  for location in locations_dict:
    latitude, longitude = locations_dict[location]
    north, south, west, east = access.create_bounding_box(latitude, longitude, 1)
    query = f"SELECT COUNT(CASE WHEN amenity = 'university' THEN 1 END) AS university, COUNT(CASE WHEN history IS NOT NULL AND history <> '' THEN 1 END) as history, COUNT(CASE WHEN leisure IS NOT NULL AND leisure <> '' THEN 1 END) as leisure,COUNT(CASE WHEN tourism IS NOT NULL AND tourism <> ''  THEN 1 END) as tourism, COUNT(CASE WHEN cuisine IS NOT NULL AND cuisine <> '' THEN 1 END) as cuisine, COUNT(CASE WHEN office IS NOT NULL AND office <> ''  THEN 1 END) as office FROM poi_counts_data WHERE LAT <{north} AND LAT >{south} AND `LONG` < {east} AND `LONG` > {west}"
    cursor.execute(query)
    result = cursor.fetchall()
    poi_df = pd.DataFrame(result, columns=["university", "history", "leisure", "tourism", "cuisine", "office"])
    pois_locations.append(poi_df.to_numpy()[0])
  pois_locations_df = pd.DataFrame(pois_locations, index=locations_dict.keys(), columns=["university", "history", "leisure", "tourism", "cuisine", "office"])
  return pois_locations_df

def pca_analysis(norm_data, locations_dict):
  pca = decomposition.PCA(n_components=2)
  pca.fit(norm_data)
  pca_transformed_data = pca.transform(norm_data)

  locations = []
  for key in locations_dict:
    locations.append(key)

  plt.figure(figsize=(10, 8))
  plt.scatter(pca_transformed_data[:, 0], pca_transformed_data[:, 1], alpha=0.5)
  for i in range(len(locations_dict)):
    plt.annotate(locations[i], (pca_transformed_data[i, 0], pca_transformed_data[i, 1]))
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')

  plt.show()

def get_average_price_paid_data(lat, lon, distance, username, password, url):
  conn = access.create_connection(username, password, url, database='ads_2024')
  cursor = conn.cursor()

  north, south, west, east = access.create_bounding_box(lat, lon, distance)

  query = f"""
  SELECT pp.price, pp.primary_addressable_object_name, pp.secondary_addressable_object_name, pp.street
  FROM pp_data AS pp
  INNER JOIN postcode_data AS pd ON pd.postcode = pp.postcode
  WHERE pd.latitude < {north} AND pd.latitude > {south} AND pd.longitude < {east} AND pd.longitude > {west}
  """
  cursor.execute(query)
  rows = cursor.fetchall()

  columns = ['price', 'primary_addressable_object_name', 'secondary_addressable_object_name', 'street']
  df1 = pd.DataFrame(rows, columns=columns)

  tags = {
    "building": True,
    "addr:postcode": True,
    "addr:housenumber": True,
    "addr:street": True,
  }
   #Place name doesn't matter as we aren't using the area or edges that get returned from the function call
  try:
    #with warnings.catch_warnings():
      #warnings.simplefilter("ignore")
    _, _, _, _, df2 = access.get_poi_info(tags, north, south, east, west, place_name= "Cambridge")
  except:
    print("no poi info")
    return 0,0,0

  cursor.close()
  conn.close()

  combined = combine_price_area(df1, df2)

  price_df = combined['price'].to_numpy()
  area_df = combined['area_sqm'].to_numpy()

  price_df = price_df[area_df != 0]
  area_df = area_df[area_df != 0]
  price_per_sqm_df = price_df / area_df

  if len(price_df) == 0 or len(area_df) == 0:
    print("df lengths not long enough")
    return 0,0,0
  av_price = np.sum(price_df)/len(price_df)
  av_area = np.sum(area_df)/len(area_df)
  av_price_per_sqm = np.sum(price_per_sqm_df)/len(price_per_sqm_df)

  return av_price, av_area, av_price_per_sqm

def combine_price_area(price_df, area_df):
  combined = []

  included_characters =  [chr(ord("A") + i) for i in range(26)]
  included_characters.append("-")
  included_characters.extend([str(i) for i in range(10)])
  area_df_dict = {}
  for index, row in area_df.iterrows():
    curr = row.to_dict()
    temp_str = curr['addr:housenumber'] + curr['addr:street'].upper()
    i = 0
    temp_str = list(temp_str)
    while i < len(temp_str):
      if temp_str[i] not in included_characters:
        temp_str.pop(i)
      else:
        i += 1
    temp_str = ''.join(temp_str)
    area_df_dict[temp_str] = curr['area_sqm']

  for index, row in price_df.iterrows():
    curr = row.to_dict()
    temp_str = curr['primary_addressable_object_name'] + curr['street'].upper()

    temp_str = list(temp_str)
    i = 0
    while i < len(temp_str):
      if temp_str[i] not in included_characters:
        temp_str.pop(i)
      else:
        i += 1
    temp_str = ''.join(temp_str)
    if temp_str in area_df_dict:
      combined.append([curr['price'], area_df_dict[temp_str]])

  combined_df = pd.DataFrame(combined, columns=['price', 'area_sqm'])

  return combined_df

def get_miniproject_df(locations_dict, username, password, url):
  conn = access.create_connection(username, password, url, database='ads_2024')
  cursor = conn.cursor()
  all_rows = []
  for location in locations_dict:
    print(location)
    latitude, longitude = locations_dict[location]
    north, south, west, east = access.create_bounding_box(latitude, longitude, 1)
    # gcd.OA21CD, gcd.lat, gcd.'long'
    query = f"""
      SELECT erd.Con, erd.Lab, erd.LD, erd.RUK, erd.Green, erd.SNP, erd.PC, erd.DUP, erd.SF, erd.SDLP, erd.UUP, erd.APNI, erd.`All other candidates`, nd.L1_L2_L3, nd.L4_L5_L6, nd.L7, nd.L8_L9, nd.L10_L11, nd.L12, nd.L13, nd.L14, nd.L15
      FROM geo_coords_data AS gcd
      INNER JOIN oa_to_constituency_data AS ocd ON ocd.OA21CD = gcd.OA21CD
      INNER JOIN election_results_data AS erd ON ocd.PCON25CD = erd.`ONSID`
      INNER JOIN nsec_data AS nd on nd.geography = ocd.OA21CD
      WHERE gcd.LAT <{north} AND gcd.LAT >{south} AND gcd.`LONG` < {east} AND gcd.`LONG` > {west}
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    curr = np.array(list(rows[0]))
    election = curr[:13]
    nsec = curr[13:]

    election = election / np.sum(election)
    nsec = nsec / np.sum(nsec)

    curr = np.concatenate((election, nsec))

    row = [location]
    row.extend(list(curr))

    av_price, av_area, av_price_per_sqm = get_average_price_paid_data(latitude,longitude, 1, username, password, url)
    row.append(av_price)
    row.append(av_area)
    row.append(av_price_per_sqm)

    all_rows.append(row)
  columns = ["location", "con", "lab", "ld", "ruk", "green", "snp", "pc", "dup", "sf", "sdlp", "uup", "apni", "other", "L1_L2_L3", "L4_L5_L6", "L7", "L8_L9", "L10_L11", "L12", "L13", "L14", "L15", "av_price", "av_area", "av_price_per_sqm"]
  df = pd.DataFrame(all_rows, columns=columns)
  df = df.set_index('location')

  df["av_price"] = df["av_price"] / df["av_price"].max()
  df["av_area"] = df["av_area"] / df["av_area"].max()
  df["av_price_per_sqm"] = df["av_price_per_sqm"] / df["av_price_per_sqm"].max()


  cursor.close()
  conn.close()
  return df

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
