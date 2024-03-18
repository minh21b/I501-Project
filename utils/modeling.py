import re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def plot_yardage_histograms(rush_data, pass_data):
    # Create histogram for rush yardages
    rush_fig = px.histogram(x=rush_data['Yards'], nbins=20, title='Rush Yardages Histogram')
    rush_fig.update_layout(yaxis_title='Frequency', xaxis_title='Yards')

    # Create histogram for pass yardages
    pass_fig = px.histogram(x=pass_data['Yards'], nbins=20, title='Pass Yardages Histogram')
    pass_fig.update_layout(yaxis_title='Frequency', xaxis_title='Yards')

    return rush_fig, pass_fig
