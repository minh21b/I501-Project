import re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error

def plot_yardage_histograms(rush_data, pass_data):
    # Create histogram for rush yardages
    rush_fig = px.histogram(x=rush_data['Yards'], nbins=20, title='Rush Yardages Histogram')
    rush_fig.update_layout(yaxis_title='Frequency', xaxis_title='Yards')

    # Create histogram for pass yardages
    pass_fig = px.histogram(x=pass_data['Yards'], nbins=20, title='Pass Yardages Histogram')
    pass_fig.update_layout(yaxis_title='Frequency', xaxis_title='Yards')

    return rush_fig, pass_fig


def train_als_model(data):
    # Constructing the user-item interaction matrix
    user_ids = data['SitID'].astype('category').cat.codes
    play_ids = data['PlayID'].astype('category').cat.codes
    ratings = data['Yards']
    interaction_matrix = coo_matrix((ratings, (user_ids, play_ids)))

    # Initializing and training the ALS model
    model = AlternatingLeastSquares(factors=100, regularization=0.01, iterations=5)
    model.fit(interaction_matrix)

    return model

def calculate_rmse(model, data):
    # Predict ratings
    user_ids = data['SitID'].astype('category').cat.codes
    play_ids = data['PlayID'].astype('category').cat.codes
    predicted_ratings = model.user_factors.dot(model.item_factors.T)

    # Flatten the interaction matrix and predicted ratings for comparison
    actual_ratings = coo_matrix((data['Yards'], (user_ids, play_ids))).toarray().flatten()
    predicted_ratings = predicted_ratings.flatten()
    # Calculate RMSE
    return np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))