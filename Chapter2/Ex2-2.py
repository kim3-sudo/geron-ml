from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Cleaning Data

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

housing = strat_train_set.drop("median_house_value")
housing_labels = strat_train_set["median_house_value"].copy()

# Let's clean some data!
housing.dropna(subset=["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis=1) # option 2
median = housing["total_bedrooms"].median() # option 3-1
housing["total_bedrooms"].fillna(median,inplace=True) # 3-2

# How do we deal with missing values?
# Let's use an imputer
imputer = SimpleImputer(strategy="median")

# Create a copy of teh data without ocean_proximity, since this is not numerical
housing_num = housing.drop("ocean_proximity", axis=1)


