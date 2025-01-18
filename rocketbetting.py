from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import requests
import openai

# The Odds API credentials and settings
API_KEY = os.getenv("ODDS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPORTS_BASE_URLS = {
    "NBA": 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds',
    "NFL": 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds',
    "MLS": 'https://api.the-odds-api.com/v4/sports/soccer_usa_mls/odds'
}

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or specify your Wix domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# OpenAI API setup
openai.api_key = OPENAI_API_KEY

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

# Function to format data for OpenAI
def format_odds_for_ai(odds_data, sport):
    game_descriptions = []
    for game in odds_data:
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        if not home_team or not away_team:
            continue

        if game.get("bookmakers"):
            bookmaker = game["bookmakers"][0]
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    outcomes = market.get("outcomes", [])
                    if len(outcomes) >= 2:
                        home_odds = next((o.get("price") for o in outcomes if o.get("name") == home_team), None)
                        away_odds = next((o.get("price") for o in outcomes if o.get("name") == away_team), None)

                        if home_odds and away_odds:
                            game_descriptions.append(f"{sport}: {home_team} vs {away_team} | Home Odds: {home_odds}, Away Odds: {away_odds}")
    return game_descriptions

# Function to generate the best pick using OpenAI
def generate_best_pick_with_ai(game_descriptions):
    if not game_descriptions:
        return {"error": "No valid games to analyze."}

    prompt = (
        "You are an AI expert in sports betting. Analyze the following games and recommend the best straight bet based "
        "on the given odds. Provide the sport, the recommended team, and a brief explanation:\n\n"
    )
    prompt += "\n".join(game_descriptions)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert sports betting assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return {"error": "Failed to generate a recommendation."}

# Endpoint to get the best straight bet
@app.get("/best-pick")
def get_best_pick():
    game_descriptions = []
    for sport, base_url in SPORTS_BASE_URLS.items():
        odds_data = fetch_odds(API_KEY, base_url)
        if odds_data:
            game_descriptions.extend(format_odds_for_ai(odds_data, sport))

    best_pick = generate_best_pick_with_ai(game_descriptions)
    return {"best_pick": best_pick}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sports Betting API!"}

if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run("rocketbetting:app", host="0.0.0.0", port=8000, reload=True)
