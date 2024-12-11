from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""
import requests
import zipfile
import pymysql
import csv
import time
import osmnx as ox
import pandas as pd
import io
import math
import ijson
import osmium
from shapely import wkt
from shapely.geometry import shape, Polygon, MultiPolygon, LineString, Point
from pyproj import Transformer
# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """
def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)

def download_open_postcode_geo_data():
    url = "https://www.getthedata.com/downloads/open_postcode_geo.csv.zip"
    zip_file_name = "/open_postcode_geo.csv.zip"
    response = requests.get(url)
    if response.status_code == 200:
        print("Downloading geo data")
        with open("." + zip_file_name, "wb") as file:
            file.write(response.content)

        print("Extracting geo data")
        file_name = "/open_postcode_geo"
        with zipfile.ZipFile("." + zip_file_name, "r") as file:            
            file.extractall("." + file_name)

def download_url_data(data_url, extract_dir):
    response = requests.get(data_url)
    if response.status_code == 200:
      with open("." + extract_dir, "wb") as file:
        file.write(response.content)
        print("Downloaded file to: ", "." + extract_dir)
    else:
      print("Failed to download the file.")

def convert_geojson_to_csv(geojson_path, csv_output_path):
    with open(geojson_path, 'r') as geojson_file:
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = None
            for feature in ijson.items(geojson_file, 'features.item'):
                properties = feature.get('properties', {})
                geometry = feature.get('geometry', {})
    
                row = {}
                row['OA21CD'] = properties.get('OA21CD', '')
                row['LSOA21CD'] = properties.get('LSOA21CD', '')
                geom_type = geometry.get('type', '')
                geom_coordinates = geometry.get('coordinates', '')
                if geom_type and geom_coordinates:
                  shape_geometry = shape({"type": geom_type, "coordinates": geom_coordinates})
                  row['geometry'] = shape_geometry.wkt
        
                if writer is None:
                  headers = list(row.keys())
                  writer = csv.DictWriter(csv_file, fieldnames=headers)
                  writer.writeheader()
        
                writer.writerow(row)

def split_csv(input_file, chunk_size, output_prefix):
    chunk = pd.read_csv(input_file, chunksize=chunk_size)
    for i, part in enumerate(chunk):
        part.to_csv(output_prefix+ "_" + str(i+1) + ".csv", index=False)
            
def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        # print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def housing_upload_join_data(conn, year):
  start_date = str(year) + "-01-01"
  end_date = str(year) + "-12-31"

  cur = conn.cursor()
  print('Selecting data for year: ' + str(year))
  cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.primary_addressable_object_name, pp.secondary_addressable_object_name, pp.street, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county, primary_addressable_object_name, secondary_addressable_object_name, street FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
  rows = cur.fetchall()

  csv_file_path = 'output_file.csv'

  # Write the rows to the CSV file
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)
  print('Storing data for year: ' + str(year))
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  conn.commit()
  print('Data stored for year: ' + str(year))
    
def create_bounding_box(latitude, longitude, distance_km):
  box_width = distance_km/111 
  north = latitude + box_width/2
  south = latitude - box_width/2
  west = longitude - box_width/(2*math.cos(math.radians(latitude)))
  east = longitude + box_width/(2*math.cos(math.radians(latitude)))
  return north, south, west, east

def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    # Create the bounding box to get north, south, west, east values
    north, south, west, east = create_bounding_box(latitude, longitude, distance_km)

    # Get the data and store in a dataframe
    pois = ox.geometries_from_bbox(north, south, east, west, tags)
    pois_df = pd.DataFrame(pois)

    dict = {}
    for key in tags:
      if key in pois_df.columns:
        # if the type of the tag is a list
        if type(tags[key]) == type([]):
          # then for each item in that list
          for tag in tags[key]:
            # sum up all values in the column that equal that item
            dict[tag] = (pois_df[key]==tag).sum()
        else:
          # if its not a list then just count all values in the list that aren't null
          dict[key] = pois_df[key].notnull().sum()
      else:
        # If its not in the columns then its 0
        dict[key] = 0

    return dict

def get_pois_df_for_locations(locations_dict, tags):
  rows = []
  # for each item in the dictionary
  for key in locations_dict:
      # get the latitude and longitude
      latitude, longitude = locations_dict[key]
      # count the places of interest according to tags
      row = count_pois_near_coordinates(latitude, longitude, tags)
      # add the location name to the dictionary
      row["location"] = key
      # append the row
      rows.append(row)

  # Create a list of all the columns
  columns = ["location"]
  for key in tags:
    if type(tags[key]) == type([]):
      for tag in tags[key]:
        columns.append(tag)
    else:
      columns.append(key)

  # Create a dataframe with the data and columns
  poi_counts_df = pd.DataFrame(rows, columns=columns)
  # Set the location to the index
  poi_counts_df.set_index("location", inplace=True)

  return poi_counts_df

def get_poi_info(tags, north, south, east, west, place_name):
  # get pois information
  pois = ox.geometries_from_bbox(north, south, east, west, tags)

  # copy before filtering
  buildings_with_addresses_pois = pois.copy()

  # filter for only places with full addresses
  for tag in tags:
    buildings_with_addresses_pois = buildings_with_addresses_pois[buildings_with_addresses_pois[tag].notnull()]

  # Make sure we are using the correct co-ordinate system then calculate area and store it
  temp = buildings_with_addresses_pois.copy()
  temp = temp.to_crs(epsg=3395)
  temp["area_sqm"] = temp.geometry.area

  # convert to a dataframe
  addresses_pois_df = pd.DataFrame(temp)

  # get the buildings without addresses
  buildings_without_addresses_pois = pois[~pois.index.isin(buildings_with_addresses_pois.index)]

  # get the graph data
  graph = ox.graph_from_bbox(north, south, east, west)

  # Retrieve nodes and edges
  nodes, edges = ox.graph_to_gdfs(graph)

  # Get place boundary related to the place name as a geodataframe
  area = ox.geocode_to_gdf(place_name)
  return area, edges, buildings_with_addresses_pois, buildings_without_addresses_pois, addresses_pois_df

def combine_data(prices_coordinates_data_result, addresses_pois_df):
  transactions_data_dic = {"primary_to_secondary": {}}
  for row in prices_coordinates_data_result:
    postcode = row[0]
    primary_addressable_object_name = row[1]

    if postcode in transactions_data_dic:
      transactions_data_dic[postcode]["primary_addressable_object_name"].append(primary_addressable_object_name)
    else:
      transactions_data_dic[postcode] = {"primary_addressable_object_name": [primary_addressable_object_name], "street": row[3], "latitude": row[4], "longitude": row[5], "price":row[6]}

  matches = []
  for i in range(len(addresses_pois_df)):
    row = addresses_pois_df.iloc[i].to_dict()
    if row["addr:postcode"] in transactions_data_dic:
      house_number = row["addr:housenumber"].split("-")
      if len(house_number) == 1:
        house_number = house_number[0].upper()
      else:
        house_number = house_number[0].upper() + " - " + house_number[1].upper()
      
      paom1 = paom2 = paom3 = paom4 = paom5 = paom6 = house_number

      if type(row["name"]) is str:
        paom4  = row["name"].upper()
        paom1 = paom4 + ", " + paom1
      if type(row["old_name"]) is str:
        paom5 = row["old_name"].upper()
        paom2 = paom5 + ", " + paom2
      if type(row["addr:housename"]) is str:
        paom6 = row["addr:housename"].upper()
        paom3 = paom6 + ", " + paom3
      
      paom_list = transactions_data_dic[row["addr:postcode"]]["primary_addressable_object_name"]
      if paom1 in paom_list or paom2 in paom_list or paom3 in paom_list or paom4 in paom_list or paom5 in paom_list or paom6 in paom_list or row["addr:housenumber"] in paom_list:
        if row["addr:street"].upper() == transactions_data_dic[row["addr:postcode"]]["street"]:
          row["longitude"] = transactions_data_dic[row["addr:postcode"]]["longitude"]
          row["latitude"] = transactions_data_dic[row["addr:postcode"]]["latitude"]
          row["price"] = transactions_data_dic[row["addr:postcode"]]["price"]
          matches.append(row)
  
  # Allows you specify order of the first columns
  columns = ["addr:housenumber","addr:street", "addr:postcode", "addr:city", "price", "area_sqm"]

  # Fills in the rest of the columns
  for key in matches[0]:
    if key not in columns:
      columns.append(key)

  price_area_df = pd.DataFrame(matches, columns=columns)
  return price_area_df

def download_census_data(code, base_dir=''):
  url = f'https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip'
  extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])

  if os.path.exists(extract_dir) and os.listdir(extract_dir):
    print(f"Files already exist at: {extract_dir}.")
    return

  os.makedirs(extract_dir, exist_ok=True)
  response = requests.get(url)
  response.raise_for_status()

  with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(extract_dir)

  print(f"Files extracted to: {extract_dir}")

def load_census_data(code, level='msoa'):
  return pd.read_csv(f'census2021-{code.lower()}/census2021-{code.lower()}-{level}.csv')

class StopProcessing(Exception):
    pass

class osmiumHandler(osmium.SimpleHandler):
    def __init__(self, tags, output_file, limit):
        super(osmiumHandler, self).__init__()
        self.data = []
        self.tags = tags
        self.additional_tags = ["addr:postcode", "addr:housenumber", "addr:street"]
        self.output_file = output_file

        columns = ['id', 'lat', 'lon']
        for key in tags:
          columns.append(key)
        for key in self.additional_tags:
            columns.append(key)
        self.columns = columns
        self.limit=limit
        self.count=0
        self.prev_area = 0
        self.prev_coords = (0, 0)

    def node(self, n):
      if self.count > self.limit:
        raise StopProcessing

      added = False
      for key in self.tags:

        if key in n.tags and not added:
          if type(self.tags[key]) == list:
            for value in self.tags[key]:
              if n.tags[key]==value:
                self.data.append(self.extract_data(n, "node"))
                self.count += 1
                added = True
                break

          else:
            self.data.append(self.extract_data(n, "node"))
            self.count += 1
            added = True

    def way(self, w):
      if self.count > self.limit:
        raise StopProcessing
      if len(w.nodes) < 2:
        return

      added = False
      for key in self.tags:
        if key in w.tags and not added:
          if type(self.tags[key]) == list:
            for value in self.tags[key]:
              if w.tags[key]==value:
                self.data.append(self.extract_data(w, "way"))
                self.count += 1
                added = True
                break

          else:
            self.data.append(self.extract_data(w, "way"))
            self.count += 1
            added = True

      valid = True
      coordinates = []
      if added == True:
        for n in w.nodes:
          if n.location.valid(): 
            coordinates.append((n.location.lat, n.location.lon,))
          else:
            valid = False
      
      if valid == True and len(coordinates) > 1:
        line = LineString(coordinates)
        centroid = line.centroid

        try:
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)
            projected_coords = [transformer.transform(lon, lat) for lon, lat in coordinates]
    
            polygon = Polygon(projected_coords)
            self.prev_area = polygon.area
        except:
            self.prev_area = None

        self.prev_coords = (centroid.x, centroid.y)
        self.data.append(self.extract_data(w, "way"))
       

    def extract_data(self, n, type_of):
      data = {}
      data['id'] = n.id
      if type_of == "node":
        data['lat'] = n.location.lat
        data['lon'] = n.location.lon
        data['area'] = 0
      else:
        data['lat'] = self.prev_coords[0]
        data['lon'] = self.prev_coords[1]
        data['area'] = self.prev_area

      for key in self.tags:
        value = self.tags[key]
        if key in n.tags:
          if (type(value) == list):
            for v in value:
              data[key] = n.tags[key]
              break
          else:
            data[key] = n.tags[key]
        else:
          data[key] = None
      
      for tag in self.additional_tags:
        if tag in n.tags:
          data[tag] = n.tags[tag]
        else:
          data[tag] = None
      return data

    def save_to_csv(self):
      with open(self.output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=self.columns)
        writer.writeheader()
        writer.writerows(self.data)
          
def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

