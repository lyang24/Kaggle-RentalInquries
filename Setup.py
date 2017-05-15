import numpy as np
from scipy import sparse
import xgboost as xgb
import pandas as pd
import re
import string
import time
from sklearn import preprocessing, pipeline, metrics, model_selection
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import feature_selection
from itertools import product

# import os
# import cv2
# import IPython.display
# from skimage import data
# from skimage import io
# from skimage.filters.rank import entropy
# from skimage.morphology import disk
# import csv
# --- tried to use opencv to locate pictures with floor plan and it overfits


import osgeo 
import fiona
import json
import geopandas as gpd
import geocoder
from scipy import sparse
from shapely.geometry import Point
from geopandas.tools import sjoin
from geopy.distance import vincenty

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline 

train_data = pd.read_json(r'..\train.json')
test_data = pd.read_json(r'..\test.json') 
train_size = train_data.shape[0]

train_data['target'] = train_data['interest_level'].apply(lambda x: 0 if x=='low' else 1 if x=='medium' else 2)
# the code blow is used for frequency encoding
train_data['low'] = train_data['interest_level'].apply(lambda x: 1 if x=='low' else 0)
train_data['medium'] = train_data['interest_level'].apply(lambda x: 1 if x=='medium' else 0)
train_data['high'] = train_data['interest_level'].apply(lambda x: 1 if x=='high' else 0)

full_data=pd.concat([train_data
                       ,test_data])