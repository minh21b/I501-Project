# Project Title: Football Recommender System

## Abstract
The National Football League (NFL) is a dynamic and complex sports environment
where strategic decision-making during playcalling significantly influences the outcomes of
games. Coaches face the challenge of choosing optimal plays based on numerous factors such
as the game situation, the opponent’s defense, their own team’s offense, tendencies, and
historical data. To enhance decision-making and improve offensive efficiency, there is a need for
a robust recommendation system that predicts the most suitable play (whether it’s run or pass)
and estimates the potential yardage gain for each play in real-time. The objective of this project
is to develop a recommendation system for play-by-play scenarios. The system should leverage
historical play data, incorporate information about game context, player statistics, and
tendencies. The model’s primary goal is to provide coaches with actionable insights to make
informed decisions about play calling, ultimately leading to improved offensive performance and
strategic advantage during games.

## Data Description
The dataset used in this project comes from https://nflsavant.com/. First, I collected the data from 2013-2015 and merged them together
to ensure that the sample size is large enough for the project. The data originally has 45 columns but there's 29 columns after cleaning.
The dataset is a publicly-available NFL play-by-play data. each row of the data includes the game time of each play recorded, the offense team,
defense team, which down, how many yards to go, and where they are on the field. Additionally, there's a 'description' column described what happened
during the play, how many yards gained from the play, which formation was the offense team, which play type occurred (rush or pass) and miscellaneous such as if the play was sacked, intercepted, penalized, or special occasions such as teams trying for a two-point conversion after a touchdown. There's a few columns I decided to remove from the original dataset include GameDate, SeriesFirstDown, NextScore, SeasonYear, columns related to plays being challenged, and two-point plays. Lastly, I added TeamID, SitID, and PlayID. TeamID is the unique number identifier for each team to make modeling easier, SitID is an arbitrary unique number identifier to identify the situation of the play (defined as YardLine + 100*ToGo+ 10000*Down + 100000*TeamID), and PlayID is also an arbitrary unique number identifier to identify the play that was called whether it's pass or rush and the pass type or rush direction of the play. 

## Algorithm Used
The NFL Football Play Recommender System uses several algorithms and techniques to provide play recommendations for the offense team. Here's an overview of the algorithms driving the web app:
### Random Forest Regressor for Play Recommendation
The modeling algorithm I used after a lot of testing was Random Forest Regressor. The Random Forest Regressor is an ensemble learning method used for regression tasks. It operates by constructing multiple decision trees during training and outputs the average prediction of the individual trees. It's a versatile algorithm that can handle both categorical and numerical features, and it's robust against overfitting which fits perfectly for this project. 
To host my web app, I used streamlit as the streaming site.
### Feature Engineering
The model considers various features including 
  - Down: Represents the current down in the play (e.g., first down, second down).
  - Distance: Indicates the distance the offense team needs to advance for a first down.
  - Yardline: Represents the field position of the offense team.
  - Offensive Team: Identifies the team on offense.
  - Play Type: Whether the play is a pass or a rush.
### Data Preprocessing
The app preprocesses the data by filtering out incomplete plays and categorizing them as either pass or rush plays. This clean data is then fed into the Random Forest Regressor for training.
### Model Prediction
When a user inputs the team, down, distance, and yard line into the web app, the trained models are used to make predictions. The model predicts the expected yardage gained for both pass and rush plays based on the input parameters.
### Play Recommendations
Based on the predictions, the app recommends the top pass and rush plays for the given situation. For pass plays, if the specific situation exists in the data, the app displays the two pass plays predicted to gain the most yards. If the situation is not found, the app recommends the two pass plays closest to the average predicted yardage for that team. The same approach is used for rush plays.
### Displaying Results
The recommended plays, along with their predicted yards gained, are displayed to the user. Users can see the details of each recommended play, such as the play type (pass or rush) and direction.

## Tools Used
In this project, for hosting the web app, I chose streamlit. To store the data remotely, I chose backblaze because it's convenient and free to use. 
In order to make this work, there's a variety of different python packages I implemented including: os, pickle, streamlit, dotenv, sklearn.ensemble, mimetypes, pandas, boto3, botocore, and a few custom modules. 
- **os**: Used for interacting with the operating system, such as accessing environment variables, managing file paths, and executing system commands.
- **pickle**: Utilized for serializing and deserializing Python objects, allowing the model to be saved and loaded for reuse.
- **streamlit**: Chosen as the framework for building the web app due to its simplicity and ease of use for creating interactive data applications in Python.
- **dotenv**: Used for loading environment variables from a .env file into the environment, enabling secure and convenient configuration of sensitive data like API keys and credentials.
- **sklearn.ensemble**: Specifically, RandomForestRegressor is used for building and training the random forest models to predict yardage gained by offensive plays in NFL games.
- **mimetypes**: Used to guess the MIME type of a file based on its filename extension, useful for working with file uploads and downloads in the web app.
- **pandas**: Employed for data manipulation and analysis, including filtering, preprocessing, and displaying data in tabular format.
- **boto3**: Used for interacting with the Backblaze B2 cloud storage service, enabling uploading, downloading, and managing files stored remotely.
- **botocore**: Dependency of boto3, providing the low-level interface to AWS services, including Backblaze B2.
- **custom modules**: the b2 module is used to connect to the virtual storage and gain access to the dataset.

## Ethical Concerns
- Recommending certain plays over others could influence game strategies, potentially impacting the outcome of games and players' careers. A solution is clearly state the limitations of the system and emphasize that the recommendations are suggestions, not guarantees of success.
- The recommender system could introduce some bias or produce low accuracy and reliability since the data was preprocessed to only considered successful passing and rushing plays which means that incompleted plays were excluded. 