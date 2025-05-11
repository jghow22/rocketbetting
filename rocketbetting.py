"""
Sports Betting API - Backend service that provides betting recommendations
using odds data and AI-powered analysis.
"""
import os
import asyncio
import json
import re
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timezone
import uuid
import traceback

# Google Sheets Integration
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Force uvicorn to use the standard asyncio loop instead of uvloop
os.environ["UVICORN_LOOP"] = "asyncio"
# Set up asyncio loop
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
# Apply nest_asyncio for nested event loops if not using uvloop
if loop.__class__.__module__.startswith("uvloop"):
    print("Detected uvloop; skipping nest_asyncio.apply()")
else:
    import nest_asyncio
    nest_asyncio.apply()
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import openai
from pydantic import BaseModel
from cachetools import TTLCache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Google Sheets Manager class
class SheetsManager:
    def __init__(self, credentials_path=None, credentials_json=None, spreadsheet_id=None):
        """Initialize Google Sheets manager"""
        self.client = None
        self.spreadsheet_id = spreadsheet_id or os.getenv("SPREADSHEET_ID")
        self.credentials_path = credentials_path
        self.credentials_json = credentials_json or os.getenv("GOOGLE_CREDENTIALS_JSON")
        
        logger.info(f"Initializing SheetsManager with spreadsheet ID: {self.spreadsheet_id}")
        
        if not self.spreadsheet_id:
            logger.error("SPREADSHEET_ID environment variable not set")
        
        if not self.credentials_path and not self.credentials_json:
            logger.error("No credentials provided - need either path or JSON content")
        elif self.credentials_json:
            logger.info("Found credentials JSON in environment")
        elif self.credentials_path:
            if not os.path.exists(self.credentials_path):
                logger.error(f"Credentials file not found at: {self.credentials_path}")
        
        # Initialize connection
        self.connect()
        
    def connect(self):
        """Connect to Google Sheets API"""
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        try:
            if self.credentials_path and os.path.exists(self.credentials_path):
                logger.info(f"Using credentials file from: {self.credentials_path}")
                credentials = ServiceAccountCredentials.from_json_keyfile_name(self.credentials_path, scope)
            elif self.credentials_json:
                logger.info("Using credentials from environment variable")
                # Parse JSON from string
                try:
                    json_dict = json.loads(self.credentials_json)
                    credentials = ServiceAccountCredentials.from_json_keyfile_dict(json_dict, scope)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse credentials JSON: {str(e)}")
                    logger.error(f"First 100 chars of JSON: {self.credentials_json[:100]}...")
                    return False
            else:
                logger.error("No valid credentials provided")
                return False
                
            self.client = gspread.authorize(credentials)
            logger.info("Successfully connected to Google Sheets")
            
            # Test connection by accessing the spreadsheet
            try:
                spreadsheet = self.client.open_by_key(self.spreadsheet_id)
                worksheet_names = [ws.title for ws in spreadsheet.worksheets()]
                logger.info(f"Successfully accessed spreadsheet. Available worksheets: {worksheet_names}")
            except Exception as e:
                logger.error(f"Could access credentials but failed to open spreadsheet: {str(e)}")
                logger.error(traceback.format_exc())
                
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Google Sheets: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def get_sheet(self, sheet_name):
        """Get a specific worksheet"""
        if not self.client or not self.spreadsheet_id:
            logger.error("Google Sheets client not properly initialized")
            return None
            
        try:
            # Open the spreadsheet
            spreadsheet = self.client.open_by_key(self.spreadsheet_id)
            worksheet = spreadsheet.worksheet(sheet_name)
            return worksheet
        except Exception as e:
            logger.error(f"Error accessing Google Sheet {sheet_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def store_game(self, game_data):
        """Store a game in the Games sheet"""
        worksheet = self.get_sheet("Games")
        if not worksheet:
            logger.error("Could not access Games worksheet")
            return False
        
        try:
            game_id = game_data.get("id", str(uuid.uuid4()))
            sport = game_data.get("sport", "Unknown")
            home_team = game_data.get("home_team", "Unknown")
            away_team = game_data.get("away_team", "Unknown")
            date = game_data.get("commence_time", "")
            
            # Get odds if available
            home_odds = "N/A"
            away_odds = "N/A"
            bookmaker = "Unknown"
            
            if game_data.get("bookmakers") and len(game_data["bookmakers"]) > 0:
                bookmaker_data = game_data["bookmakers"][0]
                bookmaker = bookmaker_data.get("title", "Unknown")
                for market in bookmaker_data.get("markets", []):
                    if market["key"] == "h2h":
                        outcomes = market.get("outcomes", [])
                        for outcome in outcomes:
                            if outcome.get("name") == home_team:
                                home_odds = outcome.get("price", "N/A")
                            elif outcome.get("name") == away_team:
                                away_odds = outcome.get("price", "N/A")
            
            status = game_data.get("status", "scheduled")
            home_score = game_data.get("home_score", "")
            away_score = game_data.get("away_score", "")
            timestamp = datetime.now().isoformat()
            
            # Append to sheet
            row = [
                game_id, sport, home_team, away_team, date, 
                home_odds, away_odds, bookmaker, status, 
                home_score, away_score, timestamp
            ]
            
            logger.info(f"Attempting to store game in sheet: {home_team} vs {away_team}")
            worksheet.append_row(row)
            logger.info(f"Successfully stored game: {home_team} vs {away_team}")
            return True
        except Exception as e:
            logger.error(f"Error storing game data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def store_prediction(self, prediction_data):
        """Store a prediction in the Predictions sheet"""
        worksheet = self.get_sheet("Predictions")
        if not worksheet:
            logger.error("Could not access Predictions worksheet")
            return False
        
        try:
            prediction_id = str(uuid.uuid4())
            pred_type = prediction_data.get("type", "Unknown")  # straight, parlay, player_prop
            sport = prediction_data.get("sport", "Overall")
            recommendation = prediction_data.get("recommendation", "Unknown")
            confidence = prediction_data.get("confidence", 0)
            explanation = prediction_data.get("explanation", "")
            created_at = datetime.now().isoformat()
            outcome = "Pending"
            user_action = ""
            
            # Append to sheet
            row = [
                prediction_id, pred_type, sport, recommendation, 
                confidence, explanation, created_at, outcome, user_action
            ]
            
            logger.info(f"Attempting to store prediction in sheet for {sport}: {recommendation}")
            worksheet.append_row(row)
            logger.info(f"Successfully stored prediction for {sport}: {recommendation}")
            return prediction_id
        except Exception as e:
            logger.error(f"Error storing prediction data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def store_player_prop(self, prop_data):
        """Store a player prop in the Player Props sheet"""
        worksheet = self.get_sheet("Player Props Sheet")
        if not worksheet:
            logger.error("Could not access Player Props Sheet worksheet")
            return False
        
        try:
            prop_id = str(uuid.uuid4())
            game_id = prop_data.get("game_id", "")
            player_name = prop_data.get("player_name", "")
            team = prop_data.get("team", "")
            prop_type = prop_data.get("prop_type", "")
            line = prop_data.get("line", "")
            over_odds = prop_data.get("over_odds", "")
            under_odds = prop_data.get("under_odds", "")
            bookmaker = prop_data.get("bookmaker", "")
            timestamp = datetime.now().isoformat()
            
            # Append to sheet
            row = [
                prop_id, game_id, player_name, team, prop_type,
                line, over_odds, under_odds, bookmaker, timestamp
            ]
            
            logger.info(f"Attempting to store player prop in sheet for {player_name}: {prop_type}")
            worksheet.append_row(row)
            logger.info(f"Successfully stored player prop for {player_name}: {prop_type}")
            return prop_id
        except Exception as e:
            logger.error(f"Error storing player prop data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def store_outcome(self, outcome_data):
        """Store an outcome in the Outcomes sheet"""
        worksheet = self.get_sheet("Outcomes")
        if not worksheet:
            logger.error("Could not access Outcomes worksheet")
            return False
        
        try:
            prediction_id = outcome_data.get("prediction_id", "")
            outcome = outcome_data.get("outcome", "")
            determined_at = datetime.now().isoformat()
            details = outcome_data.get("details", "")
            actual_result = outcome_data.get("actual_result", "")
            
            # Append to sheet
            row = [prediction_id, outcome, determined_at, details, actual_result]
            
            logger.info(f"Attempting to store outcome in sheet for prediction {prediction_id}: {outcome}")
            worksheet.append_row(row)
            
            # Also update the outcome in the Predictions sheet if possible
            try:
                predictions_sheet = self.get_sheet("Predictions")
                if predictions_sheet:
                    # Find the prediction row
                    cell = predictions_sheet.find(prediction_id)
                    if cell:
                        # Update the outcome column (column 8)
                        predictions_sheet.update_cell(cell.row, 8, outcome)
            except Exception as e:
                logger.warning(f"Could not update outcome in Predictions sheet: {str(e)}")
                
            logger.info(f"Successfully stored outcome for prediction {prediction_id}: {outcome}")
            return True
        except Exception as e:
            logger.error(f"Error storing outcome data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def store_user_interaction(self, interaction_data):
        """Store a user interaction in the User Interactions sheet"""
        worksheet = self.get_sheet("User Interactions")
        if not worksheet:
            logger.error("Could not access User Interactions worksheet")
            return False
        
        try:
            session_id = interaction_data.get("session_id", str(uuid.uuid4()))
            prediction_id = interaction_data.get("prediction_id", "")
            interaction_type = interaction_data.get("interaction_type", "")
            timestamp = datetime.now().isoformat()
            page = interaction_data.get("page", "")
            device_type = interaction_data.get("device_type", "")
            
            # Append to sheet
            row = [
                session_id, prediction_id, interaction_type, 
                timestamp, page, device_type
            ]
            
            logger.info(f"Attempting to store user interaction in sheet: {interaction_type} for {prediction_id}")
            worksheet.append_row(row)
            logger.info(f"Successfully stored user interaction: {interaction_type}")
            return True
        except Exception as e:
            logger.error(f"Error storing user interaction data: {str(e)}")
            logger.error(traceback.format_exc())
            return False

# Initialize Google Sheets integration
try:
    logger.info("About to initialize SheetsManager...")
    sheets_manager = SheetsManager()
    logger.info("Google Sheets integration initialized")
except Exception as e:
    logger.error(f"Failed to initialize Google Sheets integration: {str(e)}")
    logger.error(traceback.format_exc())  # Print the full stack trace
    sheets_manager = None

# Initialize FastAPI app
app = FastAPI(
    title="Sports Betting API",
    description="API for sports betting recommendations powered by AI",
    version="1.0.0"
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API keys from environment
API_KEY = os.getenv("ODDS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
THESPORTSDB_API_KEY = os.getenv("THESPORTSDB_API_KEY")

# Print masked API keys for debugging
if API_KEY:
    logger.info(f"Using Odds API key: {API_KEY[:5]}*****")
if OPENAI_API_KEY:
    logger.info(f"Using OpenAI API key: {OPENAI_API_KEY[:5]}*****")
if THESPORTSDB_API_KEY:
    logger.info(f"Using TheSportsDB API key: {THESPORTSDB_API_KEY[:5]}*****")

# Define available sports and their endpoints
SPORTS_BASE_URLS: Dict[str, str] = {
    "NBA": "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
    "NFL": "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
    "CFB": "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds",
    "MLS": "https://api.the-odds-api.com/v4/sports/soccer_usa_mls/odds",
    "MLB": "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
    "NHL": "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
}

# Define readable sport names
SPORT_DISPLAY_NAMES: Dict[str, str] = {
    "NBA": "Basketball (NBA)",
    "NFL": "Football (NFL)",
    "CFB": "College Football (NCAAF)",
    "MLS": "Soccer (MLS)",
    "MLB": "Baseball (MLB)",
    "NHL": "Hockey (NHL)"
}

# Create caches (TTL in seconds)
games_cache = TTLCache(maxsize=100, ttl=600)  # Cache games for 10 minutes
bets_cache = TTLCache(maxsize=100, ttl=1800)  # Cache bet recommendations for 30 minutes

def verify_api_keys():
    """Verify that required API keys are set."""
    missing_keys = []
    if not API_KEY:
        missing_keys.append("ODDS_API_KEY")
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    if not THESPORTSDB_API_KEY:
        missing_keys.append("THESPORTSDB_API_KEY")
    if missing_keys:
        missing_str = ", ".join(missing_keys)
        logger.error(f"Missing required API keys: {missing_str}")
        raise HTTPException(
            status_code=500, 
            detail=f"Server configuration error: Missing API keys ({missing_str})"
        )

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from text using regex.
    Args:
        text: String that might contain a JSON object
    Returns:
        Parsed JSON object or None if no valid JSON found
    """
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception as e:
            logger.warning(f"Failed to parse JSON with regex: {str(e)}")
            return None
    return None

def format_datetime(dt_str: str) -> str:
    """
    Format a datetime string into a user-friendly format.
    Args:
        dt_str: ISO format datetime string
    Returns:
        Formatted datetime string
    """
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%A, %B %d, %Y at %I:%M %p")
    except Exception:
        return dt_str

def evaluate_bet_value(odds: float, estimated_probability: float) -> float:
    """
    Calculate the expected value of a bet.
    Args:
        odds: Decimal odds for the bet
        estimated_probability: Our estimated probability of the bet winning (0-1)
    Returns:
        Expected value of the bet (positive is good)
    """
    return (odds * estimated_probability) - 1

def fetch_odds(
    api_key: str, 
    base_url: str, 
    markets: str = "h2h", 
    regions: str = "us"
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch odds data from The Odds API.
    Args:
        api_key: The Odds API key
        base_url: Endpoint URL for specific sport
        markets: Market types to fetch (e.g., h2h, spreads)
        regions: Region code for odds format
    Returns:
        List of game odds data or None if request fails
    """
    params = {
        "apiKey": api_key,
        "oddsFormat": "decimal",
        "markets": markets,
        "regions": regions
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Request to {base_url} failed: {str(e)}")
        return None

def fetch_player_data_thesportsdb(api_key: str, sport: str) -> List[Dict[str, Any]]:
    """
    Fetch player data from TheSportsDB API.
    Args:
        api_key: TheSportsDB API key
        sport: Sport code (e.g., NBA, NFL)
    Returns:
        List of player data or empty list if request fails
    """
    base_url = f"https://www.thesportsdb.com/api/v1/json/{api_key}/searchplayers.php"
    params = {"p": sport.lower()}
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("player", [])
    except requests.RequestException as e:
        logger.error(f"TheSportsDB request failed: {str(e)}")
        return []

def format_odds_for_ai(odds_data: List[Dict[str, Any]], sport: str) -> List[str]:
    """
    Format odds data with enhanced context for better AI analysis.
    Args:
        odds_data: List of game odds data
        sport: Sport code (e.g., NBA, NFL)
    Returns:
        List of formatted game descriptions with detailed context
    """
    game_descriptions = []
    # Get today's date for context
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for game in odds_data:
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        commence_time = game.get("commence_time", "")
        if not home_team or not away_team:
            continue
        # Format game time and determine if it's today
        game_time = ""
        is_today = False
        if commence_time:
            try:
                dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                game_time = f" on {dt.strftime('%Y-%m-%d at %H:%M UTC')}"
                is_today = dt.strftime("%Y-%m-%d") == today
            except ValueError:
                pass
        # Add additional context based on sport
        additional_context = ""
        if sport == "NBA":
            additional_context = f" | Context: NBA basketball game{', today' if is_today else ''}"
        elif sport == "NHL":
            additional_context = f" | Context: NHL hockey game{', today' if is_today else ''}"
        elif sport == "NFL":
            additional_context = f" | Context: NFL football game{', today' if is_today else ''}"
        elif sport == "MLB":
            additional_context = f" | Context: MLB baseball game{', today' if is_today else ''}"
        elif sport == "MLS":
            additional_context = f" | Context: MLS soccer match{', today' if is_today else ''}"
        elif sport == "CFB":
            additional_context = f" | Context: College football game{', today' if is_today else ''}"
            
        if game.get("bookmakers"):
            bookmaker = game["bookmakers"][0]
            bookmaker_name = bookmaker.get("title", "Unknown Bookmaker")
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    outcomes = market.get("outcomes", [])
                    if len(outcomes) >= 2:
                        home_odds = next((o.get("price") for o in outcomes if o.get("name") == home_team), None)
                        away_odds = next((o.get("price") for o in outcomes if o.get("name") == away_team), None)
                        if home_odds and away_odds:
                            # Calculate implied probabilities
                            home_implied_prob = round(100 / home_odds, 1)
                            away_implied_prob = round(100 / away_odds, 1)
                            game_descriptions.append(
                                f"{sport}: {home_team} (Home) vs {away_team} (Away){game_time} | " 
                                f"Odds: {home_team}: {home_odds} ({home_implied_prob}% implied), "
                                f"{away_team}: {away_odds} ({away_implied_prob}% implied) | "
                                f"Source: {bookmaker_name}{additional_context}"
                            )
    return game_descriptions

def format_player_odds_for_ai(odds_data: List[Dict[str, Any]], sport: str) -> List[str]:
    """
    Format player prop bet data into human-readable descriptions for AI analysis.
    Args:
        odds_data: List of player odds data
        sport: Sport code (e.g., NBA, NFL)
    Returns:
        List of formatted player bet descriptions with enhanced context
    """
    player_descriptions = []
    for game in odds_data:
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        commence_time = game.get("commence_time", "")
        matchup = f"{home_team} vs {away_team}" if home_team and away_team else ""
        # Format game time
        game_time = ""
        if commence_time:
            try:
                dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                game_time = f" on {dt.strftime('%Y-%m-%d at %H:%M UTC')}"
            except ValueError:
                pass
                
        if "player_props" not in game:
            continue
            
        for player in game["player_props"]:
            name = player.get("name")
            bet_type = player.get("type")
            odds = player.get("price")
            line = player.get("line")
            if name and bet_type and odds:
                line_str = f" ({line})" if line else ""
                implied_prob = round(100 / odds, 1) if odds else 0
                player_descriptions.append(
                    f"{sport}: {name} - {bet_type}{line_str} in {matchup}{game_time} | "
                    f"Odds: {odds} ({implied_prob}% implied probability) | "
                    f"Teams: {home_team} (Home), {away_team} (Away)"
                )
                
                # Store player prop in Google Sheets if available
                if sheets_manager:
                    try:
                        prop_data = {
                            "game_id": game.get("id", ""),
                            "player_name": name,
                            "team": home_team if name in home_team else away_team,  # crude approximation
                            "prop_type": bet_type,
                            "line": line or "",
                            "over_odds": odds if "over" in bet_type.lower() else "",
                            "under_odds": odds if "under" in bet_type.lower() else "",
                            "bookmaker": game.get("bookmakers", [{}])[0].get("title", "Unknown") if game.get("bookmakers") else "Unknown"
                        }
                        sheets_manager.store_player_prop(prop_data)
                    except Exception as e:
                        logger.error(f"Error storing player prop in Google Sheets: {str(e)}")
                        logger.error(traceback.format_exc())
                        
    return player_descriptions

def get_sport_hint(descriptions: List[str]) -> str:
    """
    Extract sport code from a list of game descriptions.
    Args:
        descriptions: List of formatted game descriptions
    Returns:
        Sport code or empty string if not found
    """
    for desc in descriptions:
        if ":" in desc:
            return desc.split(":", 1)[0].strip()
    return ""

def format_games_response(games_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format games data for frontend display.
    Args:
        games_data: Raw games data from API
    Returns:
        Formatted games data for the frontend
    """
    formatted_games = []
    for game in games_data:
        try:
            formatted_game = {
                "id": game.get("id", ""),
                "sport": game.get("sport", "Unknown"),
                "homeTeam": game.get("home_team", "Unknown"),
                "awayTeam": game.get("away_team", "Unknown"),
                "date": format_datetime(game.get("commence_time", "")),
                "raw_date": game.get("commence_time", ""),
            }
            # Add odds information if available
            if game.get("bookmakers") and len(game["bookmakers"]) > 0:
                bookmaker = game["bookmakers"][0]
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        outcomes = market.get("outcomes", [])
                        for outcome in outcomes:
                            if outcome.get("name") == game.get("home_team"):
                                formatted_game["homeOdds"] = outcome.get("price")
                            elif outcome.get("name") == game.get("away_team"):
                                formatted_game["awayOdds"] = outcome.get("price")
            formatted_games.append(formatted_game)
        except Exception as e:
            logger.warning(f"Error formatting game data: {str(e)}")
    return formatted_games

def generate_best_pick_with_ai(game_descriptions: List[str]) -> Union[Dict[str, str], Dict[str, Any]]:
    """
    Generate the best straight bet recommendation using enhanced AI analysis.
    Args:
        game_descriptions: List of formatted game descriptions
    Returns:
        Dictionary with recommendation details or error message
    """
    if not game_descriptions:
        return {"error": "No valid games to analyze."}
        
    sport_hint = get_sport_hint(game_descriptions)
    sport_display = SPORT_DISPLAY_NAMES.get(sport_hint, sport_hint) if sport_hint else ""
    sport_line = f"The sport is {sport_display}." if sport_display else ""
    
    prompt = (
        "You are an expert sports betting analyst with deep knowledge of sports statistics, team dynamics, and betting strategy. "
        "Analyze the following games and recommend ONE specific bet that offers the best value, NOT simply the best odds. "
        "\n\nIn your analysis, consider the following factors, in order of importance:"
        "\n1. Recent team performance and momentum (last 5-10 games)"
        "\n2. Head-to-head matchups between the teams this season"
        "\n3. Key player availability (injuries, rest days, etc.)"
        "\n4. Home/away performance disparities"
        "\n5. Situational advantages (back-to-back games, travel fatigue, etc.)"
        "\n6. Statistical matchups and advantages"
        "\n7. Value compared to the offered odds"
        "\n\nReturn ONLY a valid JSON object with no additional commentary. The JSON must follow EXACTLY this format:"
        '\n{"sport": "[Sport Name]", "bet": "[Team Name]", "explanation": "[Detailed reasoning with specific data points]", "confidence": [0-100]}'
        "\n\nNote: Only assign confidence scores above 80 when you have extremely strong conviction backed by multiple data points."
        "\n\n" + sport_line + "\n" + "\n".join(game_descriptions)
    )
    
    try:
        # New OpenAI API format
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=500,  # Increased token limit for more detailed analysis
            messages=[
                {"role": "system", "content": "You are an expert sports betting analyst. Respond ONLY with the valid JSON object in the exact format."},
                {"role": "user", "content": prompt}
            ]
        )
        rec_text = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            rec_json = json.loads(rec_text)
        except json.JSONDecodeError:
            rec_json = extract_json(rec_text)
            
        if not rec_json:
            logger.error(f"JSON parsing error in straight bet response: {rec_text}")
            return {"error": f"Could not parse AI recommendation"}
            
        confidence = rec_json.get('confidence', 75)  # Default to 75% if not provided
        
        result = {
                        "recommendation": f"{rec_json['bet']}",
            "explanation": rec_json['explanation'],
            "confidence": confidence,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "sport": rec_json.get('sport', sport_hint or "Unknown")
        }
        
        # Store prediction in Google Sheets
        if sheets_manager:
            try:
                logger.info("Attempting to store best pick prediction in Google Sheets")
                pred_data = {
                    "type": "straight",
                    "sport": result["sport"],
                    "recommendation": result["recommendation"],
                    "confidence": result["confidence"],
                    "explanation": result["explanation"]
                }
                prediction_id = sheets_manager.store_prediction(pred_data)
                if prediction_id:
                    result["prediction_id"] = prediction_id
                    logger.info(f"Successfully stored prediction with ID: {prediction_id}")
                else:
                    logger.warning("Failed to store prediction in Google Sheets")
            except Exception as e:
                logger.error(f"Error storing straight bet prediction in Google Sheets: {str(e)}")
                logger.error(traceback.format_exc())
                
        return result
    except Exception as e:
        logger.error(f"AI request failed for best pick: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"AI analysis failed: {str(e)}"}

def generate_best_parlay_with_ai(game_descriptions: List[str]) -> Dict[str, Any]:
    """
    Generate the best parlay bet recommendation using enhanced AI analysis.
    Args:
        game_descriptions: List of formatted game descriptions
    Returns:
        Dictionary with recommendation info or error message
    """
    if not game_descriptions:
        return {"error": "No valid games to analyze."}
        
    sport_hint = get_sport_hint(game_descriptions)
    sport_display = SPORT_DISPLAY_NAMES.get(sport_hint, sport_hint) if sport_hint else ""
    sport_line = f"The sport is {sport_display}." if sport_display else ""
    
    prompt = (
        "You are an expert sports betting analyst with deep knowledge of sports statistics, team dynamics, and betting strategy. "
        "Analyze the following games and create a 2-3 team parlay bet that offers the best value, NOT simply the highest potential payout. "
        "\n\nIn your analysis, consider the following factors for EACH game in your parlay:"
        "\n1. Recent team performance and momentum (last 5-10 games)"
        "\n2. Head-to-head matchups between the teams this season"
        "\n3. Key player availability (injuries, rest days, etc.)"
        "\n4. Home/away performance disparities"
        "\n5. Situational advantages (back-to-back games, travel fatigue, etc.)"
        "\n6. Statistical matchups and advantages"
        "\n7. Diversification of risk (avoid multiple games with similar risk profiles)"
        "\n\nReturn ONLY a valid JSON object with no additional commentary. The JSON must follow EXACTLY this format:"
        '\n{"sport": "[Sport Name]", "parlay": "[Team 1] & [Team 2] (add more teams if applicable)", "explanation": "[Detailed reasoning with specific data points for EACH pick]", "confidence": [0-100]}'
        "\n\nNote: Parlay confidence should generally be lower than straight bets due to compounding risk. Only assign confidence scores above 70 in extraordinary circumstances."
        "\n\n" + sport_line + "\n" + "\n".join(game_descriptions)
    )
    
    try:
        # New OpenAI API format
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=600,  # Increased for more detailed analysis
            messages=[
                {"role": "system", "content": "You are an expert sports betting analyst. Respond ONLY with the valid JSON object in the exact format."},
                {"role": "user", "content": prompt}
            ]
        )
        rec_text = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            rec_json = json.loads(rec_text)
        except json.JSONDecodeError:
            rec_json = extract_json(rec_text)
            
        if not rec_json:
            logger.error(f"JSON parsing error in parlay response: {rec_text}")
            return {"error": "Could not parse AI recommendation"}
            
        confidence = rec_json.get('confidence', 65)  # Default to 65% for parlays
        
        result = {
            "recommendation": f"{rec_json['parlay']}",
            "explanation": rec_json['explanation'],
            "confidence": confidence,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "sport": rec_json.get('sport', sport_hint or "Unknown")
        }
        
        # Store prediction in Google Sheets
        if sheets_manager:
            try:
                logger.info("Attempting to store parlay prediction in Google Sheets")
                pred_data = {
                    "type": "parlay",
                    "sport": result["sport"],
                    "recommendation": result["recommendation"],
                    "confidence": result["confidence"],
                    "explanation": result["explanation"]
                }
                prediction_id = sheets_manager.store_prediction(pred_data)
                if prediction_id:
                    result["prediction_id"] = prediction_id
                    logger.info(f"Successfully stored parlay prediction with ID: {prediction_id}")
                else:
                    logger.warning("Failed to store parlay prediction in Google Sheets")
            except Exception as e:
                logger.error(f"Error storing parlay prediction in Google Sheets: {str(e)}")
                logger.error(traceback.format_exc())
        
        return result
    except Exception as e:
        logger.error(f"AI request failed for best parlay: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"AI analysis failed: {str(e)}"}

def generate_best_player_bet_with_ai(player_descriptions: List[str]) -> Dict[str, Any]:
    """
    Generate the best player prop bet recommendation using enhanced AI analysis.
    Args:
        player_descriptions: List of formatted player descriptions
    Returns:
        Dictionary with recommendation info or error message
    """
    if not player_descriptions:
        return {"error": "Player prop bets are unavailable for this sport."}
        
    sport_hint = get_sport_hint(player_descriptions)
    sport_display = SPORT_DISPLAY_NAMES.get(sport_hint, sport_hint) if sport_hint else ""
    sport_line = f"The sport is {sport_display}." if sport_display else ""
    
    prompt = (
        "You are an expert sports betting analyst specializing in player performance statistics and trends. "
        "Analyze the following player prop betting options and recommend ONE specific bet that offers the best value, NOT simply the best odds. "
        "\n\nIn your analysis, consider the following factors, in order of importance:"
        "\n1. Player's recent performance trend (last 5-10 games)"
        "\n2. Player's performance against this specific opponent historically"
        "\n3. Player's role in current team strategy"
        "\n4. Matchup advantages/disadvantages (defensive matchups, etc.)"
        "\n5. Situational factors (minutes restrictions, injuries to teammates, etc.)"
        "\n6. Statistical anomalies that may regress to the mean"
        "\n7. Value compared to the offered odds"
        "\n\nReturn ONLY a valid JSON object with no additional commentary. The JSON must follow EXACTLY this format:"
        '\n{"sport": "[Sport Name]", "player_bet": "[Player Name] on [Bet Type]", "explanation": "[Detailed reasoning with specific statistical evidence]", "confidence": [0-100]}'
        "\n\nYour explanation must include specific statistical data and clear reasoning."
        "\n\n" + sport_line + "\n" + "\n".join(player_descriptions)
    )
    
    try:
        # New OpenAI API format
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=500,  # Increased for more detailed analysis
            messages=[
                {"role": "system", "content": "You are an expert sports betting analyst. Respond ONLY with the valid JSON object in the exact format."},
                {"role": "user", "content": prompt}
            ]
        )
        rec_text = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            rec_json = json.loads(rec_text)
        except json.JSONDecodeError:
            rec_json = extract_json(rec_text)
            
        if not rec_json:
            logger.error(f"JSON parsing error in player bet response: {rec_text}")
            return {"error": "Could not parse AI recommendation"}
            
        confidence = rec_json.get('confidence', 70)  # Default to 70% for player props
        
        result = {
            "recommendation": f"{rec_json['player_bet']}",
            "explanation": rec_json['explanation'],
            "confidence": confidence,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "sport": rec_json.get('sport', sport_hint or "Unknown")
        }
        
        # Store prediction in Google Sheets
        if sheets_manager:
            try:
                logger.info("Attempting to store player prop prediction in Google Sheets")
                pred_data = {
                    "type": "player_prop",
                    "sport": result["sport"],
                    "recommendation": result["recommendation"],
                    "confidence": result["confidence"],
                    "explanation": result["explanation"]
                }
                prediction_id = sheets_manager.store_prediction(pred_data)
                if prediction_id:
                    result["prediction_id"] = prediction_id
                    logger.info(f"Successfully stored player prop prediction with ID: {prediction_id}")
                else:
                    logger.warning("Failed to store player prop prediction in Google Sheets")
            except Exception as e:
                logger.error(f"Error storing player bet prediction in Google Sheets: {str(e)}")
                logger.error(traceback.format_exc())
        
        return result
    except Exception as e:
        logger.error(f"AI request failed for best player bet: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"AI analysis failed: {str(e)}"}

@app.get("/games")
async def get_games(
    sport: str = Query(None, description="Sport code (e.g., NBA, NFL)")
):
    """
    Get upcoming games and their odds.
    Args:
        sport: Optional sport code to filter results
    Returns:
        List of games with odds data or error message
    """
    # Verify API keys
    verify_api_keys()
    
    cache_key = f"games:{sport or 'all'}"
    # Check if we have cached data
    if cache_key in games_cache:
        logger.info(f"Returning cached games data for {sport or 'all sports'}")
        return games_cache[cache_key]
        
    if sport:
        sp = sport.upper()
        url = SPORTS_BASE_URLS.get(sp)
        if not url:
            return {"error": f"Sport not supported: {sport}"}
            
        data = fetch_odds(API_KEY, url)
        if not data:
            return {"error": f"No games found for {sp}."}
            
        # Store games in Google Sheets
        if sheets_manager:
            for g in data:
                g["sport"] = sp
                try:
                    result = sheets_manager.store_game(g)
                    if result:
                        logger.info(f"Successfully stored game for {sp}")
                    else:
                        logger.warning(f"Failed to store game for {sp}")
                except Exception as e:
                    logger.error(f"Error storing game in Google Sheets: {str(e)}")
                    logger.error(traceback.format_exc())
                    
        formatted_data = format_games_response(data)
        games_cache[cache_key] = formatted_data
        return formatted_data
        
    # If no sport specified, get all games
    all_games = []
    for sp, url in SPORTS_BASE_URLS.items():
        data = fetch_odds(API_KEY, url)
        if data:
            # Add sport to each game
            for g in data:
                g["sport"] = sp
                
                # Store games in Google Sheets
                if sheets_manager:
                    try:
                        result = sheets_manager.store_game(g)
                        if result:
                            logger.info(f"Successfully stored game for {sp}")
                        else:
                            logger.warning(f"Failed to store game for {sp}")
                    except Exception as e:
                        logger.error(f"Error storing game in Google Sheets: {str(e)}")
                        logger.error(traceback.format_exc())
                        
            all_games.extend(data)
            
    formatted_data = format_games_response(all_games)
    games_cache[cache_key] = formatted_data
    return formatted_data if formatted_data else {"error": "No games found."}

@app.get("/best-pick")
async def get_best_pick():
    """
    Get the best straight bet recommendation across all sports.
    Returns:
        Dictionary with best pick recommendation
    """
    # Verify API keys
    verify_api_keys()
    
    cache_key = "best_pick:all"
    # Check if we have cached data
    if cache_key in bets_cache:
        logger.info("Returning cached best pick recommendation")
        return {"best_pick": bets_cache[cache_key]}
        
    all_desc = []
    all_games = []
    for sp, url in SPORTS_BASE_URLS.items():
        data = fetch_odds(API_KEY, url)
        if data:
            # Store games in Google Sheets
            if sheets_manager:
                for game in data:
                    game["sport"] = sp
                    try:
                        result = sheets_manager.store_game(game)
                        if result:
                            logger.info(f"Successfully stored game for {sp}")
                        else:
                            logger.warning(f"Failed to store game for {sp}")
                    except Exception as e:
                        logger.error(f"Error storing game in Google Sheets: {str(e)}")
                        logger.error(traceback.format_exc())
            
            all_desc += format_odds_for_ai(data, sp)
            all_games.extend(data)
            
    result = generate_best_pick_with_ai(all_desc)
    bets_cache[cache_key] = result
    return {"best_pick": result}

@app.get("/best-parlay")
async def get_best_parlay():
    """
    Get the best parlay bet recommendation across all sports.
    Returns:
        Dictionary with best parlay recommendation
    """
    # Verify API keys
    verify_api_keys()
    
    cache_key = "best_parlay:all"
    # Check if we have cached data
    if cache_key in bets_cache:
        logger.info("Returning cached best parlay recommendation")
        return {"best_parlay": bets_cache[cache_key]}
        
    all_desc = []
    for sp, url in SPORTS_BASE_URLS.items():
        data = fetch_odds(API_KEY, url)
        if data:
            # Store games in Google Sheets if not already stored by best-pick
            if sheets_manager:
                for game in data:
                    game["sport"] = sp
                    try:
                        result = sheets_manager.store_game(game)
                        if result:
                            logger.info(f"Successfully stored game for {sp}")
                        else:
                            logger.warning(f"Failed to store game for {sp}")
                    except Exception as e:
                        logger.error(f"Error storing game in Google Sheets: {str(e)}")
                        logger.error(traceback.format_exc())
                        
            all_desc += format_odds_for_ai(data, sp)
            
    result = generate_best_parlay_with_ai(all_desc)
    bets_cache[cache_key] = result
    return {"best_parlay": result}

@app.get("/sport-best-pick")
async def get_sport_best_pick(
    sport: str = Query(..., description="Sport code (e.g., NBA, NFL)")
):
    """
    Get the best straight bet recommendation for a specific sport.
    Args:
        sport: Sport code to get recommendations for
    Returns:
        Dictionary with best pick recommendation for the sport
    """
    # Verify API keys
    verify_api_keys()
    
    cache_key = f"best_pick:{sport}"
    # Check if we have cached data
    if cache_key in bets_cache:
        logger.info(f"Returning cached best pick for {sport}")
        return {"sport_best_pick": bets_cache[cache_key]}
        
    sp = sport.upper()
    url = SPORTS_BASE_URLS.get(sp)
    if not url:
        return {"error": f"Sport not supported: {sport}"}
        
    data = fetch_odds(API_KEY, url)
    if not data:
        return {"error": f"No games found for {sp}."}
        
    # Store games in Google Sheets
    if sheets_manager:
        for game in data:
            game["sport"] = sp
            try:
                result = sheets_manager.store_game(game)
                if result:
                    logger.info(f"Successfully stored game for {sp}")
                else:
                    logger.warning(f"Failed to store game for {sp}")
            except Exception as e:
                logger.error(f"Error storing game in Google Sheets: {str(e)}")
                logger.error(traceback.format_exc())
    
    result = generate_best_pick_with_ai(format_odds_for_ai(data, sp))
    bets_cache[cache_key] = result
    return {"sport_best_pick": result}

@app.get("/sport-best-parlay")
async def get_sport_best_parlay(
    sport: str = Query(..., description="Sport code (e.g., NBA, NFL)")
):
    """
    Get the best parlay bet recommendation for a specific sport.
    Args:
        sport: Sport code to get recommendations for
    Returns:
        Dictionary with best parlay recommendation for the sport
    """
    # Verify API keys
    verify_api_keys()
    
    cache_key = f"best_parlay:{sport}"
    # Check if we have cached data
    if cache_key in bets_cache:
        logger.info(f"Returning cached best parlay for {sport}")
        return {"sport_best_parlay": bets_cache[cache_key]}
        
    sp = sport.upper()
    url = SPORTS_BASE_URLS.get(sp)
    if not url:
        return {"error": f"Sport not supported: {sport}"}
        
    data = fetch_odds(API_KEY, url)
    if not data:
        return {"error": f"No games found for {sp}."}
        
    # Store games in Google Sheets if not already stored by sport-best-pick
    if sheets_manager:
        for game in data:
            game["sport"] = sp
            try:
                result = sheets_manager.store_game(game)
                if result:
                    logger.info(f"Successfully stored game for {sp}")
                else:
                    logger.warning(f"Failed to store game for {sp}")
            except Exception as e:
                logger.error(f"Error storing game in Google Sheets: {str(e)}")
                logger.error(traceback.format_exc())
    
    result = generate_best_parlay_with_ai(format_odds_for_ai(data, sp))
    bets_cache[cache_key] = result
    return {"sport_best_parlay": result}

@app.get("/player-best-bet")
async def get_player_best_bet(
    sport: str = Query(..., description="Sport code (e.g., NBA, NFL)")
):
    """
    Get the best player prop bet recommendation for a specific sport.
    Args:
        sport: Sport code to get recommendations for
    Returns:
        Dictionary with best player bet recommendation
    """
    # Verify API keys
    verify_api_keys()
    
    if sport.upper() == "OVERALL":
        return {"best_player_bet": "Please select a specific sport for player prop bets."}
        
    cache_key = f"best_player_bet:{sport}"
    # Check if we have cached data
    if cache_key in bets_cache:
        logger.info(f"Returning cached best player bet for {sport}")
        return {"best_player_bet": bets_cache[cache_key]}
        
    sp = sport.upper()
    base_url = SPORTS_BASE_URLS.get(sp)
    if not base_url:
        return {"best_player_bet": f"Sport not supported: {sport}"}
        
    # 1) Try real player_props from Odds API
    odds_data = fetch_odds(API_KEY, base_url, markets="player_props")
    player_descriptions = format_player_odds_for_ai(odds_data or [], sp)
    
    # 2) If none, fallback to TheSportsDB
    if not player_descriptions:
        logger.info(f"No player props from odds API for {sport}, trying TheSportsDB")
        thesports = fetch_player_data_thesportsdb(THESPORTSDB_API_KEY, sp)
        for p in thesports:
            if isinstance(p, dict):
                name = p.get("strPlayer")
                pos = p.get("strPosition")
                if name and pos:
                    player_descriptions.append(f"{sp}: {name} - Position: {pos}")
    
    # 3) If still none, report unavailable
    if not player_descriptions:
        return {"best_player_bet": f"Player prop bets are unavailable for {sport}."}
        
    result = generate_best_player_bet_with_ai(player_descriptions)
    bets_cache[cache_key] = result
    return {"best_player_bet": result}

@app.get("/available-sports")
async def get_available_sports():
    """
    Get list of available sports with their codes and display names.
    Returns:
        List of sport codes and display names
    """
    return [
        {"code": "Overall", "name": "All Sports"},
        *[{"code": code, "name": display} for code, display in SPORT_DISPLAY_NAMES.items()]
    ]

@app.get("/clear-cache")
async def clear_cache():
    """
    Clear all cached data to force fresh API calls and recommendations.
    Returns:
        Confirmation message
    """
    games_cache.clear()
    bets_cache.clear()
    return {"message": "Cache cleared successfully. Next requests will fetch fresh data."}

@app.get("/")
def read_root():
    """
    Root endpoint - welcome message.
    Returns:
        Welcome message
    """
    return {
        "message": "Welcome to the Sports Betting API!",
        "version": "1.1.0",
        "endpoints": [
            "/games", 
            "/best-pick", 
            "/best-parlay", 
            "/sport-best-pick", 
            "/sport-best-parlay", 
            "/player-best-bet",
            "/available-sports",
            "/clear-cache",
            "/test-sheets-connection",  # New endpoint for testing sheets connection
            "/track-interaction"
        ]
    }

# New endpoint for tracking user interactions
@app.get("/track-interaction")
async def track_interaction(
    prediction_id: str = Query(None, description="ID of the prediction viewed"),
    interaction_type: str = Query("view", description="Type of interaction (view, bookmark, share)"),
    page: str = Query(None, description="Page where interaction occurred"),
    device_type: str = Query(None, description="Device type (desktop, mobile, tablet)")
):
    """
    Track user interactions with predictions for analytics.
    Args:
        prediction_id: ID of the prediction interacted with
        interaction_type: Type of interaction
        page: Page where interaction occurred
        device_type: Device type
    Returns:
        Success message
    """
    if sheets_manager:
        try:
            logger.info(f"Tracking interaction: {interaction_type} for prediction {prediction_id}")
            interaction_data = {
                "prediction_id": prediction_id,
                "interaction_type": interaction_type,
                "page": page,
                "device_type": device_type
            }
            result = sheets_manager.store_user_interaction(interaction_data)
            if result:
                logger.info("Successfully recorded interaction")
                return {"message": "Interaction recorded successfully"}
            else:
                logger.warning("Failed to record interaction")
        except Exception as e:
            logger.error(f"Error recording interaction: {str(e)}")
            logger.error(traceback.format_exc())
            
    return {"message": "Failed to record interaction"}

# Add a test endpoint for Google Sheets
@app.get("/test-sheets-connection")
async def test_sheets_connection():
    """Simple test endpoint to verify Google Sheets connection"""
    if not sheets_manager or not sheets_manager.client:
        return {
            "status": "error",
            "message": "No active Google Sheets connection",
            "spreadsheet_id": os.getenv("SPREADSHEET_ID"),
            "has_credentials_json": bool(os.getenv("GOOGLE_CREDENTIALS_JSON")),
            "has_credentials_path": bool(os.getenv("GOOGLE_CREDENTIALS_PATH"))
        }
    
    try:
        # Test spreadsheet access
        spreadsheet = sheets_manager.client.open_by_key(sheets_manager.spreadsheet_id)
        worksheet_names = [ws.title for ws in spreadsheet.worksheets()]
        
        # Try to write to a worksheet
        test_sheet = spreadsheet.worksheet("Games")
        row = ["TEST", "Connection Test", datetime.now().isoformat()]
        test_sheet.append_row(row)
        
        return {
            "status": "success",
            "message": "Successfully connected to Google Sheets and wrote test data",
            "worksheets": worksheet_names
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error testing connection: {str(e)}"
        }

if __name__ == "__main__":
    uvicorn.run("rocketbetting:app", host="0.0.0.0", port=8000, reload=True)
