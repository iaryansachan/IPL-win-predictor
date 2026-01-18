import streamlit as st
import pickle
import pandas as pd

# -------------------------------
# Teams and Cities (Dropdown lists)
# -------------------------------
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# -------------------------------
# Load trained model
# -------------------------------
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title("IPL Win Predictor 🏏")

# -------------------------------
# Helper: Overs to balls
# Example: 14.2 overs means 14 overs + 2 balls = 86 balls
# -------------------------------
def overs_to_balls(overs):
    overs = float(overs)
    whole_overs = int(overs)
    balls = int(round((overs - whole_overs) * 10))  # 0.2 -> 2 balls
    return whole_overs * 6 + balls

# -------------------------------
# Input UI
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select the batting team", sorted(teams))

with col2:
    bowling_team = st.selectbox("Select the bowling team", sorted(teams))

selected_city = st.selectbox("Select host city", sorted(cities))

target = st.number_input("Target", min_value=0, step=1)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input("Score", min_value=0, step=1)

with col4:
    overs = st.text_input("Overs completed (e.g. 14.2)", value="0.0")

with col5:
    wickets_out = st.number_input("Wickets out", min_value=0, max_value=10, step=1)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Probability"):
    if batting_team == bowling_team:
        st.error("Batting team and Bowling team cannot be the same ❌")
    else:
        try:
            balls_bowled = overs_to_balls(overs)
            balls_left = 120 - balls_bowled

            # validations
            if balls_left <= 0:
                st.error("Overs completed should be less than 20.0 ❌")
            else:
                runs_left = target - score
                wickets_remaining = 10 - wickets_out

                # Run rates
                overs_done = balls_bowled / 6
                crr = score / overs_done if overs_done > 0 else 0
                rrr = (runs_left * 6) / balls_left

                # Input dataframe (must match training features)
                input_df = pd.DataFrame({
                    'batting_team': [batting_team],
                    'bowling_team': [bowling_team],
                    'city': [selected_city],
                    'runs_left': [runs_left],
                    'balls_left': [balls_left],
                    'wickets': [wickets_remaining],
                    'total_runs_x': [target],
                    'crr': [crr],
                    'rrr': [rrr]
                })

                result = pipe.predict_proba(input_df)

                loss = result[0][0]
                win = result[0][1]

                st.subheader("Prediction Result ✅")
                st.header(f"{batting_team} - {round(win * 100, 2)}%")
                st.header(f"{bowling_team} - {round(loss * 100, 2)}%")

        except Exception as e:
            st.error(f"Invalid Overs Format. Use like 14.2, 10.0 etc. Error: {e}")
