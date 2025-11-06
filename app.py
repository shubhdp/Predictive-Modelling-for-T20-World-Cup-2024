import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

@st.cache_resource
def load_models():
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    dl_model = load_model('dl_model.h5')
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    return rf_model, dl_model, scaler, le

rf_model, dl_model, scaler, le = load_models()

df = pd.read_csv("t20_worldcup_team_features_large_1000.csv")
teams = sorted(df['Team'].unique())

st.title("T20 World Cup 2025 Match Predictor")

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Select Team 1", teams)
with col2:
    team2 = st.selectbox("Select Team 2", teams)

model_type = st.radio("Select Prediction Model", 
                      ["Random Forest", "Deep Learning"])

if st.button("Predict Winner"):
    if team1 == team2:
        st.error("Please select two different teams!")
    else:
        df_2025 = df[df['Year'] == 2025].copy()
        
        if 'Team_encoded' not in df_2025.columns:
            df_2025['Team_encoded'] = le.transform(df_2025['Team'])
        
        features = [
            'Team_encoded', 'ICC_Rank', 'Batting_Rating', 'Bowling_Rating',
            'Win_Rate_Last5', 'Group_Stage_Wins', 'Avg_Strike_Rate', 'Avg_Economy_Rate'
        ]
        
        missing_features = [f for f in features if f not in df_2025.columns]
        if missing_features:
            st.error(f"Missing features in data: {', '.join(missing_features)}")
            st.stop()
        
        try:
            team1_data = df_2025[df_2025['Team'] == team1][features].iloc[0]
            team2_data = df_2025[df_2025['Team'] == team2][features].iloc[0]
        except IndexError:
            st.error("One or both selected teams not found in 2025 data")
            st.stop()
        
        X = np.array([team1_data, team2_data])
        
        if model_type == "Random Forest":
            probs = rf_model.predict_proba(X)[:, 1]  
        else:

            X_scaled = scaler.transform(X)
            probs = dl_model.predict(X_scaled)[:, 1]
        
        team1_prob = probs[0]
        team2_prob = probs[1]
        
        if team1_prob > team2_prob:
            winner = team1
            winner_prob = team1_prob
            loser_prob = team2_prob
        else:
            winner = team2
            winner_prob = team2_prob
            loser_prob = team1_prob
       
        st.success(f"Predicted Winner: {winner} ({winner_prob:.1%} chance)")
        st.subheader("Win Probabilities:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=team1, value=f"{team1_prob:.1%}")
        with col2:
            st.metric(label=team2, value=f"{team2_prob:.1%}")

        chart_data = pd.DataFrame({
            'Team': [team1, team2],
            'Win Probability': [team1_prob, team2_prob]
        })
        chart_data['Team'] = pd.Categorical(chart_data['Team'], categories=[team1, team2], ordered=True)
        chart_data = chart_data.sort_values('Team')
        st.bar_chart(chart_data.set_index('Team'))