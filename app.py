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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

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

# ------------------------------------------------------
#                         APP
# ------------------------------------------------------
# ------------------------------
# PART 0 : Overview
# ------------------------------
st.write(
'''
# Data Overview and Visualization
We pull data from our Backblaze storage bucket, and render it in Streamlit.
A quantitative question of the data I plan to use in my final project is whether pass plays on average gain more yardage or rush plays.
'''
)
data = get_data()
# ------------------------------
# PART 1 : Filter Data
# ------------------------------
rush_data = data[data['IsRush'] == 1]
pass_data = data[data['IsPass'] == 1]

# ------------------------------
# PART 2 : Modeling
# ------------------------------

# Train pass play model
X_pass = pass_data[['SitID', 'PlayID']]
y_pass = pass_data['Yards']
# Train rush play model
X_rush = rush_data[['SitID', 'PlayID']]
y_rush = rush_data['Yards']

pass_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
pass_regressor.fit(X_pass, y_pass)
rush_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rush_regressor.fit(X_rush, y_rush)

rush_data.loc[:, 'Predicted Yards'] = rush_regressor.predict(X_rush)
best_rush_plays = rush_data.sort_values(by='Predicted Yards', ascending=False)
pass_data.loc[:, 'Predicted Yards'] = pass_regressor.predict(X_pass)
best_pass_plays = pass_data.sort_values(by='Predicted Yards', ascending=False)

@st.cache_data
def getPlay(team, down, distance, yardline):
    teamId = data.loc[data['OffenseTeam'] == team, 'TeamID'].values[0]
    SitId = yardline + 100 * distance + 10000 * down + 100000 * teamId
    SitId = 1711083
    if SitId in best_pass_plays['SitID'].values:
        best_passes = best_pass_plays[best_pass_plays['SitID'] == SitId]
        best_passes = best_passes.nlargest(2, 'Predicted Yards')
        print('First Pass Choice: ', best_passes.iloc[0]['PassType'])
        print('Predicted Gain: ', round(float(best_passes.iloc[0]['Predicted Yards'])))
        print('Second Pass Choice: ', best_passes.iloc[1]['PassType'])
        print('Predicted Gain: ', round(float(best_passes.iloc[1]['Predicted Yards'])))

    if SitId in best_rush_plays['SitID'].values:
        best_rushes = best_rush_plays[best_rush_plays['SitID'] == SitId]
        best_rushes = best_rushes.nlargest(2, 'Predicted Yards')
        print('First Rush Choice: ', best_rushes.iloc[0]['RushDirection'])
        print('Predicted Gain: ', round(float(best_rushes.iloc[0]['Predicted Yards'])))
        print('Second Rush Choice: ', best_rushes.iloc[1]['RushDirection'])
        print('Predicted Gain: ', round(float(best_rushes.iloc[1]['Predicted Yards'])))

st.write(
'''
**Sample Demo of the recommender system:**
''')

# Streamlit app
def main():
    st.title("Football Play Recommender")
    
    # Input fields for the parameters
    team = st.text_input("Team")
    down = st.number_input("Down", min_value=1, max_value=4, value=1, step=1)
    distance = st.number_input("Distance", min_value=1, value=5, step=1)
    yardline = st.number_input("Yardline", min_value=1, value=95, step=1)
    
    # Start button to execute the getPlay function
    if st.button("Start"):
        # Call the getPlay function with the input parameters
        result = getPlay(team, down, distance, yardline)
        st.write("Output:")
        st.write(result)

if __name__ == "__main__":
    main()

