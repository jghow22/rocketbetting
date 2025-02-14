from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import requests
import openai

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

# Debugging: Print API keys (masked for security)
print(f"Using OpenAI API key: {os.getenv('OPENAI_API_KEY')[:5]}*****")
print(f"Using Odds API key: {os.getenv('ODDS_API_KEY')[:5]}*****")

API_KEY = os.getenv("ODDS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPORTS_BASE_URLS = {
    "NBA": 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds',
    "NFL": 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds',
    "MLS": 'https://api.the-odds-api.com/v4/sports/soccer_usa_mls/odds',
}

openai.api_key = OPENAI_API_KEY

# Function to fetch odds
def fetch_odds(api_key, base_url, markets="h2h"):
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': markets,
        'oddsFormat': 'decimal',
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

# Function to format game odds for AI
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

# Function to format player props for AI
def format_player_odds_for_ai(odds_data, sport):
    player_descriptions = []
    for game in odds_data:
        if "player_props" not in game:
            continue
        for player in game.get("player_props", []):
            player_name = player.get("name")
            bet_type = player.get("type")
            odds = player.get("price")
            if player_name and bet_type and odds:
                player_descriptions.append(f"{sport}: {player_name} - {bet_type} | Odds: {odds}")
    return player_descriptions

# Function to generate best pick with AI
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
        return {"error": f"Failed to generate a recommendation: {e}"}

# Function to generate best parlay with AI
def generate_best_parlay_with_ai(game_descriptions):
    if not game_descriptions:
        return {"error": "No valid games to analyze."}

    prompt = (
        "You are an AI expert in sports betting. Create the best parlay bet from the following games. "
        "Include the sport, the teams involved, and explain why this parlay is a strong choice:\n\n"
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
        return {"error": f"Failed to generate a recommendation: {e}"}

# Function to generate best player bet with AI
def generate_best_player_bet_with_ai(player_descriptions):
    if not player_descriptions:
        return {"error": "No valid player bets to analyze."}

    prompt = (
        "You are an AI expert in sports betting. Analyze the following player-specific betting options and recommend the "
        "best individual player bet based on the given odds. Provide the sport, the player's name, and a brief explanation:\n\n"
    )
    prompt += "\n".join(player_descriptions)

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
        return {"error": f"Failed to generate a recommendation: {e}"}

# Endpoint: Fetch game schedule
@app.get("/games")
def get_games():
    all_games = []
    for sport, base_url in SPORTS_BASE_URLS.items():
        odds_data = fetch_odds(API_KEY, base_url)
        if odds_data:
            all_games.extend(odds_data)
    return all_games if all_games else {"error": "No games found."}

# Endpoint: Best overall straight bet
@app.get("/best-pick")
def get_best_pick():
    game_descriptions = []
    for sport, base_url in SPORTS_BASE_URLS.items():
        odds_data = fetch_odds(API_KEY, base_url)
        if odds_data:
            game_descriptions.extend(format_odds_for_ai(odds_data, sport))
    return {"best_pick": generate_best_pick_with_ai(game_descriptions)}

# Endpoint: Best overall parlay bet
@app.get("/best-parlay")
def get_best_parlay():
    game_descriptions = []
    for sport, base_url in SPORTS_BASE_URLS.items():
        odds_data = fetch_odds(API_KEY, base_url)
        if odds_data:
            game_descriptions.extend(format_odds_for_ai(odds_data, sport))
    return {"best_parlay": generate_best_parlay_with_ai(game_descriptions)}

# NBA Endpoints
@app.get("/nba-best-pick")
def get_nba_best_pick():
    nba_odds_data = fetch_odds(API_KEY, SPORTS_BASE_URLS["NBA"])
    if not nba_odds_data:
        return {"error": "No NBA games found."}
    game_descriptions = format_odds_for_ai(nba_odds_data, "NBA")
    return {"nba_best_pick": generate_best_pick_with_ai(game_descriptions)}

@app.get("/nba-best-parlay")
def get_nba_best_parlay():
    nba_odds_data = fetch_odds(API_KEY, SPORTS_BASE_URLS["NBA"])
    if not nba_odds_data:
        return {"error": "No NBA games found."}
    game_descriptions = format_odds_for_ai(nba_odds_data, "NBA")
    return {"nba_best_parlay": generate_best_parlay_with_ai(game_descriptions)}

# NFL Endpoints
@app.get("/nfl-best-pick")
def get_nfl_best_pick():
    nfl_odds_data = fetch_odds(API_KEY, SPORTS_BASE_URLS["NFL"])
    if not nfl_odds_data:
        return {"error": "No NFL games found."}
    game_descriptions = format_odds_for_ai(nfl_odds_data, "NFL")
    return {"nfl_best_pick": generate_best_pick_with_ai(game_descriptions)}

@app.get("/nfl-best-parlay")
def get_nfl_best_parlay():
    nfl_odds_data = fetch_odds(API_KEY, SPORTS_BASE_URLS["NFL"])
    if not nfl_odds_data:
        return {"error": "No NFL games found."}
    game_descriptions = format_odds_for_ai(nfl_odds_data, "NFL")
    return {"nfl_best_parlay": generate_best_parlay_with_ai(game_descriptions)}

# MLS Endpoints
@app.get("/mls-best-pick")
def get_mls_best_pick():
    mls_odds_data = fetch_odds(API_KEY, SPORTS_BASE_URLS["MLS"])
    if not mls_odds_data:
        return {"error": "No MLS games found."}
    game_descriptions = format_odds_for_ai(mls_odds_data, "MLS")
    return {"mls_best_pick": generate_best_pick_with_ai(game_descriptions)}

@app.get("/mls-best-parlay")
def get_mls_best_parlay():
    mls_odds_data = fetch_odds(API_KEY, SPORTS_BASE_URLS["MLS"])
    if not mls_odds_data:
        return {"error": "No MLS games found."}
    game_descriptions = format_odds_for_ai(mls_odds_data, "MLS")
    return {"mls_best_parlay": generate_best_parlay_with_ai(game_descriptions)}

# Endpoint: Best player-specific bet (across all sports)
@app.get("/player-best-bet")
def get_player_best_bet():
    player_descriptions = []
    for sport, base_url in SPORTS_BASE_URLS.items():
        odds_data = fetch_odds(API_KEY, base_url, markets="player_props")
        print(f"Raw player data for {sport}: {odds_data}")  # Debugging log
        if odds_data:
            formatted_data = format_player_odds_for_ai(odds_data, sport)
            player_descriptions.extend(formatted_data)
            print(f"Formatted player data for {sport}: {formatted_data}")  # Debugging log

    if not player_descriptions:
        print("No player-specific data found.")
        return {"error": "Player picks unavailable at this time. This could be due to API data limitations or lack of active player props."}

    best_player_bet = generate_best_player_bet_with_ai(player_descriptions)
    if isinstance(best_player_bet, dict) and "error" in best_player_bet:
        print("Error generating player bet:", best_player_bet["error"])
        return {"error": best_player_bet["error"]}
    
    print("Generated player bet:", best_player_bet)  # Debugging log
    return {"best_player_bet": best_player_bet}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sports Betting API!"}

if __name__ == "__main__":
    uvicorn.run("rocketbetting:app", host="0.0.0.0", port=8000, reload=True)
