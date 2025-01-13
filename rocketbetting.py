from fastapi import FastAPI
import uvicorn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import requests

# The Odds API credentials and settings
API_KEY = os.getenv("ODDS_API_KEY")
NBA_BASE_URL = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'

# Define FastAPI app
app = FastAPI()

# Define model and scaler
nba_model = None
nba_scaler = None

# Function to fetch odds
def fetch_odds(api_key, base_url):
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'decimal',
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch odds: {response.status_code} - {response.text}")
        return None

# Function to extract features
def extract_features(odds_data):
    games = []
    for game in odds_data:
        home_team = game.get('home_team')
        away_team = game.get('away_team')

        if not home_team or not away_team:
            continue

        if game.get('bookmakers'):
            bookmaker = game['bookmakers'][0]
            for market in bookmaker.get('markets', []):
                if market['key'] == 'h2h':
                    outcomes = market.get('outcomes', [])
                    if len(outcomes) >= 2:
                        home_odds = next((o.get('price') for o in outcomes if o.get('name') == home_team), None)
                        away_odds = next((o.get('price') for o in outcomes if o.get('name') == away_team), None)

                        if not home_odds or not away_odds:
                            continue

                        home_prob = 1 / home_odds
                        away_prob = 1 / away_odds
                        total_prob = home_prob + away_prob
                        home_prob /= total_prob
                        away_prob /= total_prob

                        games.append({
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_prob': home_prob,
                            'away_prob': away_prob,
                            'label': 1 if home_prob > away_prob else 0,
                        })
    return pd.DataFrame(games)

# Endpoint to train NBA model
@app.get("/train")
def train_model():
    global nba_model, nba_scaler
    nba_odds = fetch_odds(API_KEY, NBA_BASE_URL)
    if nba_odds:
        nba_data = extract_features(nba_odds)
        if not nba_data.empty:
            nba_features = nba_data[['home_prob', 'away_prob']]
            nba_target = nba_data['label']
            X_train, X_test, y_train, y_test = train_test_split(nba_features, nba_target, test_size=0.2, random_state=42)
            nba_scaler = StandardScaler()
            X_train_scaled = nba_scaler.fit_transform(X_train)
            nba_model = LogisticRegression()
            nba_model.fit(X_train_scaled, y_train)
            return {"message": "NBA model trained successfully!"}
        else:
            return {"error": "No valid NBA data extracted from odds."}
    return {"error": "Failed to fetch NBA odds data."}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sports Betting API!"}

if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run("rocketbetting:app", host="0.0.0.0", port=8000, reload=True)
