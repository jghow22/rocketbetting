import os
import asyncio
import json
import re
import requests
import openai
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from requests_html import AsyncHTMLSession
from bs4 import BeautifulSoup

# Force uvicorn to use the standard asyncio loop instead of uvloop
os.environ["UVICORN_LOOP"] = "asyncio"
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
if loop.__class__.__module__.startswith("uvloop"):
    print("Detected uvloop; skipping nest_asyncio.apply()")
else:
    import nest_asyncio
    nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Print API keys (masked)
print(f"Using OpenAI API key: {os.getenv('OPENAI_API_KEY')[:5]}*****")
print(f"Using Odds API key: {os.getenv('ODDS_API_KEY')[:5]}*****")
print(f"Using TheSportsDB API key: {os.getenv('THESPORTSDB_API_KEY')[:5]}*****")

# Retrieve API keys from environment
API_KEY = os.getenv("ODDS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
THESPORTSDB_API_KEY = os.getenv("THESPORTSDB_API_KEY")

# IMPORTANT: Ensure you have pinned openai==0.28 in your requirements.txt for compatibility.
openai.api_key = OPENAI_API_KEY

# Base URLs for game odds
SPORTS_BASE_URLS = {
    "NBA": "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
    "NFL": "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
    "MLS": "https://api.the-odds-api.com/v4/sports/soccer_usa_mls/odds",
}

# ---------------------------
# Helper Functions
# ---------------------------
def extract_json(text):
    """Extract a JSON object from text using regex."""
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None

def fetch_odds(api_key, base_url, markets="h2h", regions="us"):
    params = {
        "apiKey": api_key,
        "oddsFormat": "decimal",
    }
    if markets:
        params["markets"] = markets
    if regions is not None:
        params["regions"] = regions
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request to {base_url} failed with status code {response.status_code} and response: {response.text}")
    return None

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
                            game_descriptions.append(
                                f"{sport}: {home_team} vs {away_team} | Home Odds: {home_odds}, Away Odds: {away_odds}"
                            )
    return game_descriptions

def format_player_odds_for_ai(odds_data, sport):
    player_descriptions = []
    for game in odds_data:
        if "player_props" in game:
            for prop in game["player_props"]:
                player_name = prop.get("name")
                bet_type = prop.get("type")
                odds = prop.get("price")
                if player_name and bet_type and odds:
                    player_descriptions.append(f"{sport}: {player_name} - {bet_type} | Odds: {odds}")
    return player_descriptions

def get_sport_hint(descriptions):
    for desc in descriptions:
        if ":" in desc:
            return desc.split(":", 1)[0].strip()
    return ""

def generate_best_pick_with_ai(game_descriptions):
    if not game_descriptions:
        return {"error": "No valid games to analyze."}
    sport_hint = get_sport_hint(game_descriptions)
    sport_line = f"The sport is {sport_hint}." if sport_hint else ""
    prompt = (
        "You are an expert sports betting assistant. Analyze the following games and choose one specific straight bet that you consider the best. "
        "DO NOT choose a bet solely based on high odds. Evaluate matchup context, team performance, injuries, and risk factors. "
        "Return ONLY a valid JSON object with no additional commentary. The JSON must follow EXACTLY this format:\n"
        '{"sport": "[Sport Name]", "bet": "[Team Name]", "explanation": "[Your reasoning]"}\n\n'
        + sport_line + "\n"
    )
    prompt += "\n".join(game_descriptions)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=300,
            messages=[
                {"role": "system", "content": "You are an expert sports betting assistant. Respond ONLY with the valid JSON object in the exact format."},
                {"role": "user", "content": prompt}
            ]
        )
        rec_text = response["choices"][0]["message"]["content"].strip()
        try:
            rec_json = json.loads(rec_text)
        except Exception as e:
            rec_json = extract_json(rec_text)
            if not rec_json:
                return {"error": f"JSON parsing error in straight bet response: {e}. Response was: {rec_text}"}
        return f"Sport: {rec_json['sport']} - Bet: {rec_json['bet']}. Explanation: {rec_json['explanation']}"
    except Exception as e:
        return {"error": f"Failed to generate straight bet recommendation: {e}"}

def generate_best_parlay_with_ai(game_descriptions):
    if not game_descriptions:
        return {"error": "No valid games to analyze."}
    sport_hint = get_sport_hint(game_descriptions)
    sport_line = f"The sport is {sport_hint}." if sport_hint else ""
    prompt = (
        "You are an expert sports betting assistant. Analyze the following games and choose one specific parlay bet that you consider the best. "
        "DO NOT choose a parlay solely because it includes bets with the highest odds. Consider matchups, risk distribution, and overall value. "
        "Return ONLY a valid JSON object with no additional commentary. The JSON must follow EXACTLY this format:\n"
        '{"sport": "[Sport Name]", "parlay": "[Team 1] & [Team 2] (add more teams if applicable)", "explanation": "[Your reasoning]"}\n\n'
        + sport_line + "\n"
    )
    prompt += "\n".join(game_descriptions)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=300,
            messages=[
                {"role": "system", "content": "You are an expert sports betting assistant. Respond ONLY with the valid JSON object in the exact format."},
                {"role": "user", "content": prompt}
            ]
        )
        rec_text = response["choices"][0]["message"]["content"].strip()
        try:
            rec_json = json.loads(rec_text)
        except Exception as e:
            rec_json = extract_json(rec_text)
            if not rec_json:
                return {"error": f"JSON parsing error in parlay response: {e}. Response was: {rec_text}"}
        return f"Sport: {rec_json['sport']} - Parlay: {rec_json['parlay']}. Explanation: {rec_json['explanation']}"
    except Exception as e:
        return {"error": f"Failed to generate parlay recommendation: {e}"}

def generate_best_player_bet_with_ai(player_descriptions):
    if not player_descriptions:
        return {"error": "Player prop bets are unavailable for this sport."}
    sport_hint = get_sport_hint(player_descriptions)
    sport_line = f"The sport is {sport_hint}." if sport_hint else ""
    prompt = (
        "You are an expert sports betting assistant. Analyze the following player-specific betting options and choose one specific player bet that you consider the best. "
        "DO NOT select a bet solely based on the highest odds; consider player performance, matchup context, and overall value. "
        "Return ONLY a valid JSON object with no additional commentary. The JSON must follow EXACTLY this format:\n"
        '{"sport": "[Sport Name]", "player_bet": "[Player Name] on [Bet Type]", "explanation": "[Your reasoning]"}\n\n'
        + sport_line + "\n"
    )
    prompt += "\n".join(player_descriptions)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=300,
            messages=[
                {"role": "system", "content": "You are an expert sports betting assistant. Respond ONLY with the valid JSON object in the exact format."},
                {"role": "user", "content": prompt}
            ]
        )
        rec_text = response["choices"][0]["message"]["content"].strip()
        try:
            rec_json = json.loads(rec_text)
        except Exception as e:
            rec_json = extract_json(rec_text)
            if not rec_json:
                return {"error": f"JSON parsing error in player bet response: {e}. Response was: {rec_text}"}
        return f"Sport: {rec_json['sport']} - Player Bet: {rec_json['player_bet']}. Explanation: {rec_json['explanation']}"
    except Exception as e:
        return {"error": f"Failed to generate player bet recommendation: {e}"}

def fetch_player_data_thesportsdb(api_key, sport):
    """
    Fetch player data from TheSportsDB for the given sport.
    Adjust the 'p' query parameter or endpoint as needed for your target sport/league.
    """
    base_url = f"https://www.thesportsdb.com/api/v1/json/{api_key}/searchplayers.php"
    params = {"p": "a"}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("player", [])
    else:
        print(f"TheSportsDB request failed: {response.status_code}, {response.text}")
    return []

# ---------------------------
# Endpoints
# ---------------------------

@app.get("/games")
def get_games(sport: str = Query(None, description="Sport code (e.g., NBA, NFL, MLS)")):
    if sport:
        base_url = SPORTS_BASE_URLS.get(sport.upper())
        if base_url:
            odds_data = fetch_odds(API_KEY, base_url)
            if odds_data:
                for game in odds_data:
                    game["sport"] = sport.upper()
                return odds_data
            else:
                return {"error": f"No games found for {sport}."}
        else:
            return {"error": "Sport not supported."}
    else:
        all_games = []
        for sport_key, base_url in SPORTS_BASE_URLS.items():
            odds_data = fetch_odds(API_KEY, base_url)
            if odds_data:
                for game in odds_data:
                    game["sport"] = sport_key
                all_games.extend(odds_data)
        return all_games if all_games else {"error": "No games found."}

@app.get("/best-pick")
def get_best_pick():
    # Overall best straight bet across all sports
    all_game_descriptions = []
    for sport, base_url in SPORTS_BASE_URLS.items():
        odds_data = fetch_odds(API_KEY, base_url)
        if odds_data:
            all_game_descriptions.extend(format_odds_for_ai(odds_data, sport))
    return {"best_pick": generate_best_pick_with_ai(all_game_descriptions)}

@app.get("/best-parlay")
def get_best_parlay():
    # Overall best parlay bet across all sports
    all_game_descriptions = []
    for sport, base_url in SPORTS_BASE_URLS.items():
        odds_data = fetch_odds(API_KEY, base_url)
        if odds_data:
            all_game_descriptions.extend(format_odds_for_ai(odds_data, sport))
    return {"best_parlay": generate_best_parlay_with_ai(all_game_descriptions)}

# Sports-specific endpoints using query parameter

@app.get("/sport-best-pick")
def get_sport_best_pick(sport: str = Query(..., description="Sport code (e.g., NBA, NFL, MLS)")):
    base_url = SPORTS_BASE_URLS.get(sport.upper())
    if not base_url:
        return {"error": "Sport not supported."}
    odds_data = fetch_odds(API_KEY, base_url)
    if not odds_data:
        return {"error": f"No {sport} games found."}
    game_descriptions = format_odds_for_ai(odds_data, sport.upper())
    return {"sport_best_pick": generate_best_pick_with_ai(game_descriptions)}

@app.get("/sport-best-parlay")
def get_sport_best_parlay(sport: str = Query(..., description="Sport code (e.g., NBA, NFL, MLS)")):
    base_url = SPORTS_BASE_URLS.get(sport.upper())
    if not base_url:
        return {"error": "Sport not supported."}
    odds_data = fetch_odds(API_KEY, base_url)
    if not odds_data:
        return {"error": f"No {sport} games found."}
    game_descriptions = format_odds_for_ai(odds_data, sport.upper())
    return {"sport_best_parlay": generate_best_parlay_with_ai(game_descriptions)}

@app.get("/player-best-bet")
async def get_player_best_bet(sport: str = Query("NBA", description="Sport code (e.g., NBA, NFL, MLS)")):
    player_descriptions = []
    if sport.upper() == "NBA":
        thesportsdb_data = fetch_player_data_thesportsdb(THESPORTSDB_API_KEY, sport.upper())
        if thesportsdb_data:
            for player in thesportsdb_data:
                if isinstance(player, dict):
                    name = player.get("strPlayer")
                    position = player.get("strPosition")
                    if name and position:
                        desc = f"{sport.upper()}: {name} - Position: {position}"
                        player_descriptions.append(desc)
        if not player_descriptions:
            # Fallback NBA data if TheSportsDB returns no data
            if sport.upper() == "NBA":
                player_descriptions = [
                    "NBA: LeBron James - Position: Forward",
                    "NBA: Stephen Curry - Position: Guard",
                    "NBA: Kevin Durant - Position: Forward",
                    "NBA: Giannis Antetokounmpo - Position: Forward",
                    "NBA: James Harden - Position: Guard"
                ]
    else:
        # For other sports, you may add similar logic
        player_descriptions = [f"{sport.upper()}: No player data available."]
    best_player_bet = generate_best_player_bet_with_ai(player_descriptions)
    return {"best_player_bet": best_player_bet}

# ---------------------------
# Root Endpoint
# ---------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sports Betting API!"}

if __name__ == "__main__":
    uvicorn.run("rocketbetting:app", host="0.0.0.0", port=8000, reload=True)
