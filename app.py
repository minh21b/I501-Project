import os
import pickle
from utils.b2 import B2
#from utils.modeling import *
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
# import warnings
# warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

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
data = get_data()
# ------------------------------
# PART 1 : Filter Data
# ------------------------------
@st.cache_data
def prep_data(data):
    rush_data = data[data['IsRush'] == 1]
    pass_data = data[data['IsPass'] == 1]
    # Train pass play model
    X_pass = pass_data[['SitID', 'PlayID']]
    y_pass = pass_data['Yards']
    # Train rush play model
    X_rush = rush_data[['SitID', 'PlayID']]
    y_rush = rush_data['Yards']
    return X_pass, y_pass, X_rush, y_rush, rush_data, pass_data

X_pass, y_pass, X_rush, y_rush, rush_data, pass_data = prep_data(data)
# ------------------------------
# PART 2 : Modeling
# ------------------------------
@st.cache_data
def get_model(X_pass, y_pass, X_rush, y_rush, rush_data, pass_data):
    pass_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    pass_regressor.fit(X_pass, y_pass)
    rush_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rush_regressor.fit(X_rush, y_rush)
    rush_data.loc[:, 'Predicted Yards'] = rush_regressor.predict(X_rush)
    best_rush_plays = rush_data.sort_values(by='Predicted Yards', ascending=False)
    pass_data.loc[:, 'Predicted Yards'] = pass_regressor.predict(X_pass)
    best_pass_plays = pass_data.sort_values(by='Predicted Yards', ascending=False)
    return best_rush_plays, best_pass_plays

best_rush_plays, best_pass_plays = get_model(X_pass, y_pass, X_rush, y_rush, rush_data, pass_data)

# ------------------------------
# PART 2 : Building the recommender system
# ------------------------------
@st.cache_data
def getPlay(team, down, distance, yardline):
    teamId = data.loc[data['OffenseTeam'] == team, 'TeamID'].values[0]
    SitId = yardline + 100 * distance + 10000 * down + 100000 * teamId
    if SitId in best_pass_plays['SitID'].values:
        best_passes = best_pass_plays[best_pass_plays['SitID'] == SitId]
        average_yards = best_passes['Yards'].mean()
        nearest_plays = best_pass_plays.iloc[(best_pass_plays['Predicted Yards'] - average_yards).abs().argsort()[:2]]
        best_passes = best_passes.nlargest(2, 'Predicted Yards')
        #sometimes the model doesn't recommend 2 plays, doing it this way ensures there's always at least 2 available plays
        best_passes = pd.concat([best_passes, nearest_plays])
        st.write('First Pass Choice: ', best_passes.iloc[0]['PassType'])
        st.write('Predicted Gain: ', round(float(best_passes.iloc[0]['Predicted Yards'])))
        st.write('Second Pass Choice: ', best_passes.iloc[1]['PassType'])
        st.write('Predicted Gain: ', round(float(best_passes.iloc[1]['Predicted Yards'])))
    
    else:
        best_passes = best_pass_plays[best_pass_plays['TeamID'] == teamId]
        average_yards = best_passes['Yards'].mean()
        # Select two plays nearest to the average
        nearest_plays = best_pass_plays.iloc[(best_pass_plays['Predicted Yards'] - average_yards).abs().argsort()[:2]]
        st.write('First Pass Choice: ', nearest_plays.iloc[0]['PassType'])
        st.write('Predicted Gain: ', round(float(nearest_plays.iloc[0]['Predicted Yards'])))
        st.write('Second Pass Choice: ', nearest_plays.iloc[1]['PassType'])
        st.write('Predicted Gain: ', round(float(nearest_plays.iloc[1]['Predicted Yards'])))

    if SitId in best_rush_plays['SitID'].values:
        best_rushes = best_rush_plays[best_rush_plays['SitID'] == SitId]
        average_yards = best_rushes['Yards'].mean()
        best_rushes = best_rushes.nlargest(2, 'Predicted Yards')
        nearest_plays = best_rush_plays.iloc[(best_rush_plays['Predicted Yards'] - average_yards).abs().argsort()[:2]]
        #sometimes the model doesn't recommend 2 plays, doing it this way ensures there's always at least 2 available plays
        best_rushes = pd.concat([best_rushes, nearest_plays])
        st.write('First Rush Choice: ', best_rushes.iloc[0]['RushDirection'])
        st.write('Predicted Gain: ', round(float(best_rushes.iloc[0]['Predicted Yards'])))
        st.write('Second Rush Choice: ', best_rushes.iloc[1]['RushDirection'])
        st.write('Predicted Gain: ', round(float(best_rushes.iloc[1]['Predicted Yards'])))

    else:
        best_rushes = best_rush_plays[best_rush_plays['TeamID'] == teamId]
        average_yards = best_rushes['Yards'].mean()
        # Select two plays nearest to the average
        nearest_plays = best_rush_plays.iloc[(best_rush_plays['Predicted Yards'] - average_yards).abs().argsort()[:2]]
        st.write('First Pass Choice: ', nearest_plays.iloc[0]['RushDirection'])
        st.write('Predicted Gain: ', round(float(nearest_plays.iloc[0]['Predicted Yards'])))
        st.write('Second Pass Choice: ', nearest_plays.iloc[1]['RushDirection'])
        st.write('Predicted Gain: ', round(float(nearest_plays.iloc[1]['Predicted Yards'])))


# Streamlit app
def main():
    st.title("Football Play Recommender")

    st.write(
    """
    
    This app is designed to help you find the best offensive plays for your team based on various game situations.
    Whether you're a coach, a player, or just a fan curious about football strategy, this tool can provide valuable
    insights into play selection.
    """
    )
    # Instructions
    st.write(
        """
        ## How to Use
        
        1. **Select Your Team**: Choose your team from the dropdown list. This represents the team on offense.
    
        2. **Enter Game Situation**: 
        - **Down**: Choose the current down from 1 to 4.
        - **Distance**: Enter the distance needed to gain a new set of downs. This is typically measured in yards.
        - **Yardline**: Enter the yardline where the ball is currently placed. This is the distance from the opponent's end zone, measured in yards.

        3. **Click Start**: After entering the game situation, click the "Start" button to see the recommended plays.

        4. **Review Recommendations**: The app will display the top two recommended passing and rushing plays based on the provided game situation. For passing plays, it will show the type of pass and the predicted gain in yards. For rushing plays, it will show the direction of the run and the predicted gain in yards.

        5. **Adjust and Try Again**: If you want to explore different scenarios, simply adjust the game situation parameters and click "Start" again to see new recommendations.
        
        Enjoy using the Football Play Recommender to optimize your team's offensive strategy and gain an edge on the field!
        """
    )
    
    # Input fields for the parameters
    team = st.selectbox("Team", data['OffenseTeam'].dropna().unique())
    down = st.number_input("Down", min_value=1, max_value=4, value=1, step=1)
    distance = st.number_input("Distance", min_value=1, value=10, step=1, max_value = 99)
    yardline = st.number_input("Yardline", min_value=1, value=25, step=1, max_value = 99)
    
    # Start button to execute the getPlay function
    if st.button("Start"):
        # Call the getPlay function with the input parameters
        getPlay(team, down, distance, yardline)

if __name__ == "__main__":
    main()
    # st.write(
    # '''
    # ----------------------------
    # **Issues I've ran into:** \n
    # So far there isn't too many issues I've ran to, most of them have to deal with the modeling part of the project where some models aren't
    # producing the results I expected or some models that I unfortunately cannot attempt to use due to technical difficulties with the environment
    # and I couldn't figure them out. \n
    # ----------------------------
    # **Next steps:** \n
    # I'd say this is pretty much complete in terms of building the model and stuff. All I have left is to actually make an app (decorating the web page, etc)
    # ''')
