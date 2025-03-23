import os
# Force uvicorn to use the standard asyncio loop instead of uvloop
os.environ["UVICORN_LOOP"] = "asyncio"

import asyncio
try:
    # If a loop is already running and is not uvloop, we can apply nest_asyncio.
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
if loop.__class__.__module__.startswith("uvloop"):
    print("Detected uvloop; skipping nest_asyncio.apply()")
else:
    import nest_asyncio
    nest_asyncio.apply()

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import openai
import json
import re  # For extracting JSON via regex
import time

# For asynchronous scraping using requests_html
from requests_html import AsyncHTMLSession
from bs4 import BeautifulSoup

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
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

# Retrieve API keys from environment
API_KEY = os.getenv("ODDS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Endpoints for standard game odds
SPORTS_BASE_URLS = {
    "NBA": "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
    "NFL": "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
    "MLS": "https://api.the-odds-api.com/v4/sports/soccer_usa_mls/odds",
}

openai.api_key = OPENAI_API_KEY

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
    # Parse player prop data from the API response (if available)
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
    print("Straight bet prompt:", prompt)
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
        print("Straight bet raw response:", rec_text)
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
    print("Parlay prompt:", prompt)
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
        print("Parlay raw response:", rec_text)
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
    print("Player bet prompt:", prompt)
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
        print("Player bet raw response:", rec_text)
        try:
            rec_json = json.loads(rec_text)
        except Exception as e:
            rec_json = extract_json(rec_text)
            if not rec_json:
                return {"error": f"JSON parsing error in player bet response: {e}. Response was: {rec_text}"}
        return f"Sport: {rec_json['sport']} - Player Bet: {rec_json['player_bet']}. Explanation: {rec_json['explanation']}"
    except Exception as e:
        return {"error": f"Failed to generate player bet recommendation: {e}"}

# --- Asynchronous scraping for player props from DraftKings using requests_html ---
async def scrape_draftkings_player_props(sport: str):
    """Scrape player prop data from DraftKings for the given sport.
       Adjust the URL and selectors based on DraftKings’ current structure.
       Ensure compliance with DraftKings' terms of service.
    """
    url = None
    if sport.upper() == "NBA":
        url = "https://sportsbook.draftkings.com/play-lines/nba"
    elif sport.upper() == "NFL":
        url = "https://sportsbook.draftkings.com/play-lines/nfl"
    elif sport.upper() == "MLS":
        url = "https://sportsbook.draftkings.com/play-lines/mls"
    if not url:
        return []
    
    session = AsyncHTMLSession()
    try:
        r = await session.get(url)
        await r.html.arender(timeout=60)
        html = r.html.html
    except Exception as e:
        print(f"Error scraping DraftKings for {sport}: {e}")
        return []
    
    soup = BeautifulSoup(html, "html.parser")
    player_props = []
    
    # Try using new selectors based on data-testid attributes (update these as needed)
    prop_divs = soup.find_all("div", attrs={"data-testid": "prop-market"})
    if not prop_divs:
        # Fallback to previous class-based selectors
        prop_divs = soup.find_all("div", class_="sportsbook-prop")
    
    for div in prop_divs:
        try:
            # Try new selectors first
            name_el = div.find("span", attrs={"data-testid": "player-name"})
            if name_el:
                name = name_el.get_text(strip=True)
            else:
                name_el = div.find("span", class_="sportsbook-prop__player-name")
                if not name_el:
                    continue
                name = name_el.get_text(strip=True)
            
            label_el = div.find("span", attrs={"data-testid": "prop-label"})
            if label_el:
                prop_type = label_el.get_text(strip=True)
            else:
                label_el = div.find("span", class_="sportsbook-prop__label")
                prop_type = label_el.get_text(strip=True) if label_el else "N/A"
            
            odds_el = div.find("span", attrs={"data-testid": "prop-odds"})
            if odds_el:
                odds = odds_el.get_text(strip=True)
            else:
                odds_el = div.find("span", class_="sportsbook-prop__odds")
                odds = odds_el.get_text(strip=True) if odds_el else "N/A"
            
            player_props.append(f"{sport.upper()}: {name} - {prop_type} | Odds: {odds}")
        except Exception as e:
            print(f"Error parsing a prop div for {sport}: {e}")
            continue
    return player_props

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
    game_descriptions = []
    for sport, base_url in SPORTS_BASE_URLS.items():
        odds_data = fetch_odds(API_KEY, base_url)
        if odds_data:
            game_descriptions.extend(format_odds_for_ai(odds_data, sport))
    return {"best_pick": generate_best_pick_with_ai(game_descriptions)}

@app.get("/best-parlay")
def get_best_parlay():
    game_descriptions = []
    for sport, base_url in SPORTS_BASE_URLS.items():
        odds_data = fetch_odds(API_KEY, base_url)
        if odds_data:
            game_descriptions.extend(format_odds_for_ai(odds_data, sport))
    return {"best_parlay": generate_best_parlay_with_ai(game_descriptions)}

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

@app.get("/player-best-bet")
async def get_player_best_bet():
    player_descriptions = []
    # First, attempt to get player prop data from the API endpoint with player prop markets.
    for sport, base_url in SPORTS_BASE_URLS.items():
        odds_data = fetch_odds(API_KEY, base_url, markets="player_points,player_assists,player_rebounds,player_steals,player_blocks", regions="us")
        print(f"Raw player data for {sport}: {odds_data}")
        if odds_data:
            formatted_data = format_player_odds_for_ai(odds_data, sport)
            player_descriptions.extend(formatted_data)
            print(f"Formatted player data for {sport}: {formatted_data}")
    # If no player data is returned from the API, then try scraping DraftKings.
    if not player_descriptions:
        print("No player-specific data from API; attempting to scrape DraftKings.")
        for sport in SPORTS_BASE_URLS.keys():
            scraped_data = await scrape_draftkings_player_props(sport)
            if scraped_data:
                player_descriptions.extend(scraped_data)
                print(f"Scraped player data for {sport}: {scraped_data}")
    if not player_descriptions:
        print("No player-specific data found.")
        return {"best_player_bet": "Player prop bets are unavailable for this sport."}
    best_player_bet = generate_best_player_bet_with_ai(player_descriptions)
    if isinstance(best_player_bet, dict) and "error" in best_player_bet:
        print("Error generating player bet:", best_player_bet["error"])
        return {"best_player_bet": best_player_bet["error"]}
    print("Generated player bet:", best_player_bet)
    return {"best_player_bet": best_player_bet}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sports Betting API!"}

if __name__ == "__main__":
    uvicorn.run("rocketbetting:app", host="0.0.0.0", port=8000, reload=True)
