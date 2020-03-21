import os
import tarfile
import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Fetch the data from a GitHub repository
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
  os.makedirs(housing_path, exist_ok=True)
  tgz_path = os.path.join(housing_path, "housing.tgz")
  urllib.request.urlretrieve(housing_url, tgz_path)
  housing_tgz = tarfile.open(tgz_path)
  housing_tgz.extractall(path=housing_path)
  housing_tgz.close()

# Load the data using pandas
def load_housing_data(housing_path=HOUSING_PATH):
  csv_path = os.path.join(housing_path, "housing.csv")
  return pd.read_csv(csv_path)

# Print the header of the pandas dataframe
# Make the dataframe
housingdata = pd.read_csv("housing.csv")
# Iterate over the columns
for col in housingdata.columns:
  print(col)

%matplotlib inline
housing.hist(bins=50, figsize=(20,15))
plt.show()

def split_train_test(data, test_ratio):
  shuffled_indices = np.random.permutation(len(data))
  test_set_size = int(len(data) * test_ratio)
  test_indices = shuffled_indices[:test_set_size]
  train_indices = shuffled_indices[test_set_size:]
  return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
len(train_set)
len(test_set)

def test_set_check(identifier, test_ratio):
  return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
  ids = data[id_column]
  in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
  return data.loc[-in_test_set], data.loc[in_test_set]

# Let's use the row index as the ID
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# What about the latitude and longitude
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# Use scikit-learn to do the same thing
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Cut that dataframe into a bunch of different chunks for a histogram
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()

# Try to do stratified sampling on the dataset using scikit-learn's StratifiedShuffleSplit class
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]
  
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# Now, remove the income_cat attribute so data is back to orig. state
for set_ in (strat_train_set, strat_test_set):
  set_.drop("income_cat", axis=1, inplace=True)

#######################################
# Create a copy so we can play without harming original dataset
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
# That kind of looks like California, but let's highlight high density areas
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# Now, let's look at housing prices, color using jet colormap
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), coloarbar=True,
            )
plt.legend()

# Let's look for correlations now
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# We can plot using pandas, too
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))

# Correlation between median_house_value and median_income looks promising...
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
