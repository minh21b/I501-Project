import os
import pickle
from utils.b2 import B2
from utils.modeling import *
import streamlit as st
from dotenv import load_dotenv
import numpy as np
from implicit.als import AlternatingLeastSquares
from sklearn.metrics import mean_squared_error
from scipy.sparse import coo_matrix
import category_encoders as ce
import pandas as pd
import pymc as pm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
REMOTE_DATA = 'pbp.csv'

# ------------------------------------------------------
#                        CONFIG
# ------------------------------------------------------
load_dotenv()

# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_KEYID'],
        secret_key=os.environ['B2_APPKEY'])

# ------------------------------------------------------
#                        CACHING
# ------------------------------------------------------
@st.cache_data
def get_data():
    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df_pbp = b2.get_df(REMOTE_DATA)

    return df_pbp


