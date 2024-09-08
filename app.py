import pandas as pd
import pickle
import streamlit as st

import base64

# Function to convert image to base64
def get_base64_of_bin_file(image_file):
    with open(image_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to add background image
def add_bg_from_local(image_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{image_path}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Convert image to base64 and add as background
#bg_image = get_base64_of_bin_file("C:/Users/91862/Downloads/ipl-2021-ix7zwgff29ylomuf.jpg")

bg_image = get_base64_of_bin_file("C:/Users/91862/Downloads/MS-Dhoni-1-3.jpg")

add_bg_from_local(bg_image)

# App Title with emoji
st.title('🏏 IPL Match Win Predictor')

# Subtitle
st.subheader('Predict the probability of a team winning an IPL match!')

# Highlight background colors
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Teams and Cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders','Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']
city = ['Hyderabad', 'Delhi', 'Johannesburg', 'Mohali', 'Abu Dhabi', 'Dharamsala', 'Mumbai',
        'Pune', 'Port Elizabeth', 'Bangalore', 'Durban', 'Nagpur', 'Chennai', 'Jaipur',
        'Chandigarh', 'Centurion', 'Cuttack', 'Sharjah', 'Kolkata', 'Ranchi', 'Cape Town',
        'Ahmedabad', 'Visakhapatnam', 'Raipur', 'Indore', 'Bloemfontein', 'Bengaluru',
        'Kimberley', 'East London']

# Load model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Columns for inputs
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('🏏 Select the batting team:', teams)
with col2:
    bowling_team = st.selectbox('🥎 Select the bowling team:', teams)

city = st.selectbox('🌍 Select the match city:', city)
target = st.number_input('🎯 Target Score', min_value=0)

# Another row for other match details
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wicket = st.number_input('Wickets Out', min_value=0, max_value=10)

# Predict Button
if st.button('Predict Outcome'):
    # Calculations for match conditions
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wicket
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Creating input dataframe for model prediction
    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team],
                             'city': [city], 'runs_left': [runs_left], 'balls left': [balls_left],
                             'wicket': [wickets], 'total_runs_x': [target],
                             'crr': [crr], 'rrr': [rrr]})

    # Model prediction
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Display results with some style
    st.success(f"🏏 **{batting_team}: {round(win * 100)}% chance to win!**")
    st.error(f"🥎 **{bowling_team}: {round(loss * 100)}% chance to win!**")

    # Progress bar for win probability
    st.progress(int(win * 100))
