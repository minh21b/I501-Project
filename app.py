import os
import pickle

import streamlit as st
from dotenv import load_dotenv

from utils.b2 import B2
from utils.modeling import *

# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
REMOTE_DATA = 'pbp-2023.csv'

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
filtered_data = data[data['IsIncomplete'] != 1]
rush_plays = filtered_data[filtered_data['IsRush'] == 1]
pass_plays = filtered_data[filtered_data['IsPass'] == 1]

st.write(
'''
**Your filtered data (only plays resulted in a non incompletion for simplicity purposes):**
''')

st.dataframe(filtered_data)

# ------------------------------
# PART 2 : Plot
# ------------------------------
rush_fig, pass_fig = plot_yardage_histograms(rush_plays, pass_plays)
st.plotly_chart(rush_fig)
st.plotly_chart(pass_fig)