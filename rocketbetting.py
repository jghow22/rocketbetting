"""
Sports Betting API - Backend service that provides betting recommendations
using odds data and AI-powered analysis.
"""
import os
import asyncio
import json
import re
import logging
import time
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timezone
import uuid
import traceback
from functools import lru_cache
import random
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

# Pydantic models for request/response validation
class OutcomeData(BaseModel):
    prediction_id: str
    outcome: str  # "Win", "Loss", "Push", or "Pending"
    details: Optional[str] = None
    actual_result: Optional[str] = None

# Google Sheets Manager class
class SheetsManager:
    def __init__(self, credentials_path=None, credentials_json=None, spreadsheet_id=None):
        """Initialize Google Sheets manager"""
        self.client = None
        self.spreadsheet_id = spreadsheet_id or os.getenv("SPREADSHEET_ID")
        self.credentials_path = credentials_path
        self.credentials_json = credentials_json or os.getenv("GOOGLE_CREDENTIALS_JSON")
        self.spreadsheet = None
        self.worksheet_cache = {}
        
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
            
            # Cache the spreadsheet reference for future use
            try:
                self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
                worksheet_names = [ws.title for ws in self.spreadsheet.worksheets()]
                logger.info(f"Successfully accessed spreadsheet. Available worksheets: {worksheet_names}")
                
                # Pre-cache all worksheet references to reduce API calls
                for ws_name in worksheet_names:
                    self.worksheet_cache[ws_name] = self.spreadsheet.worksheet(ws_name)
            except Exception as e:
                logger.error(f"Could access credentials but failed to open spreadsheet: {str(e)}")
                logger.error(traceback.format_exc())
                return False
                
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Google Sheets: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def get_sheet(self, sheet_name):
        """Get a specific worksheet with caching and retries"""
        if not self.client or not self.spreadsheet_id:
            logger.error("Google Sheets client not properly initialized")
            return None
            
        # Check if we already have this worksheet cached
        if sheet_name in self.worksheet_cache:
            return self.worksheet_cache[sheet_name]
            
        # Try to get the worksheet with retries
        max_retries = 3
        retry_delay = 1 # Start with 1 second delay
        for attempt in range(max_retries):
            try:
                if not self.spreadsheet:
                    logger.info("Spreadsheet reference not found, attempting to open it")
                    self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
                    
                logger.info(f"Attempting to access worksheet: {sheet_name}")
                worksheet = self.spreadsheet.worksheet(sheet_name)
                logger.info(f"Successfully accessed worksheet: {sheet_name}")
                
                # Cache the worksheet for future use
                self.worksheet_cache[sheet_name] = worksheet
                return worksheet
            except gspread.exceptions.APIError as e:
                if "429" in str(e) and attempt < max_retries - 1: # Rate limit error
                    logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2 # Exponential backoff
                else:
                    logger.error(f"Error accessing Google Sheet {sheet_name}: {str(e)}")
                    logger.error(traceback.format_exc())
                    return None
            except Exception as e:
                logger.error(f"Error accessing Google Sheet {sheet_name}: {str(e)}")
                logger.error(traceback.format_exc())
                return None
                
        logger.error(f"Failed to access worksheet {sheet_name} after {max_retries} attempts")
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
        except gspread.exceptions.APIError as e:
            if "429" in str(e): # Rate limit error
                logger.warning(f"Rate limit hit when storing game, will retry later")
                return False
            logger.error(f"Error storing game data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
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
            pred_type = prediction_data.get("type", "Unknown") # straight, parlay, player_prop
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
            
            # Also create a pending outcome record
            try:
                outcome_data = {
                    "prediction_id": prediction_id,
                    "outcome": "Pending",
                    "details": f"Automatically created for {pred_type} bet",
                    "actual_result": ""
                }
                self.store_outcome(outcome_data)
                logger.info(f"Created initial pending outcome record for prediction {prediction_id}")
            except Exception as e:
                logger.error(f"Error creating initial outcome record: {str(e)}")
            
            return prediction_id
        except gspread.exceptions.APIError as e:
            if "429" in str(e): # Rate limit error
                logger.warning(f"Rate limit hit when storing prediction, will retry later")
                # In a real system, you might queue this for later retry
                return False
            logger.error(f"Error storing prediction data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        except Exception as e:
            logger.error(f"Error storing prediction data: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def store_player_prop(self, prop_data):
        """Store a player prop in the Player Props Sheet"""
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
        except gspread.exceptions.APIError as e:
            if "429" in str(e): # Rate limit error
                logger.warning(f"Rate limit hit when storing player prop, will retry later")
                return False
            logger.error(f"Error storing player prop data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
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
        except gspread.exceptions.APIError as e:
            if "429" in str(e): # Rate limit error
                logger.warning(f"Rate limit hit when storing outcome, will retry later")
                return False
            logger.error(f"Error storing outcome data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        except Exception as e:
            logger.error(f"Error storing outcome data: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def store_user_interaction(self, interaction_data):
        """Store a user interaction in the User Interactions Sheet"""
        worksheet = self.get_sheet("User Interactions Sheet")
        if not worksheet:
            logger.error("Could not access User Interactions Sheet worksheet")
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
        except gspread.exceptions.APIError as e:
            if "429" in str(e): # Rate limit error
                logger.warning(f"Rate limit hit when storing user interaction, will retry later")
                return False
            logger.error(f"Error storing user interaction data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        except Exception as e:
            logger.error(f"Error storing user interaction data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def update_metrics(self, metrics_data):
        """Store metrics data in the Metrics sheet"""
        worksheet = self.get_sheet("Metrics")
        if not worksheet:
            logger.error("Could not access Metrics worksheet")
            return False
        
        try:
            timestamp = datetime.now().isoformat()
            metric_type = metrics_data.get("type", "Unknown")
            value = metrics_data.get("value", 0)
            sport = metrics_data.get("sport", "Overall")
            details = metrics_data.get("details", "")
            
            # Append to sheet
            row = [timestamp, metric_type, value, sport, details]
            logger.info(f"Attempting to store metrics in sheet: {metric_type}")
            worksheet.append_row(row)
            logger.info(f"Successfully stored metrics: {metric_type}")
            return True
        except gspread.exceptions.APIError as e:
            if "429" in str(e): # Rate limit error
                logger.warning(f"Rate limit hit when storing metrics, will retry later")
                return False
            logger.error(f"Error storing metrics data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        except Exception as e:
            logger.error(f"Error storing metrics data: {str(e)}")
            logger.error(traceback.format_exc())
            return False

# Initialize Google Sheets integration
try:
    logger.info("About to initialize SheetsManager...")
    sheets_manager = SheetsManager()
    logger.info("Google Sheets integration initialized")
except Exception as e:
    logger.error(f"Failed to initialize Google Sheets integration: {str(e)}")
    logger.error(traceback.format_exc()) # Print the full stack trace
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
games_cache = TTLCache(maxsize=100, ttl=600) # Cache games for 10 minutes
bets_cache = TTLCache(maxsize=100, ttl=1800) # Cache bet recommendations for 30 minutes

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
        data = resp.json()
        
        # Check if the response contains player data
        if not data or "player" not in data or not data["player"]:
            logger.warning(f"TheSportsDB API returned no player data for {sport}")
            return []
            
        return data.get("player", [])
    except requests.RequestException as e:
        logger.error(f"TheSportsDB request failed: {str(e)}")
        return []
    except ValueError as e:
        logger.error(f"Could not parse TheSportsDB response: {str(e)}")
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
            logger.info(f"No player_props found in game data for {matchup}")
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
                
                                # Store player prop in Google Sheets
                if sheets_manager:
                    try:
                        logger.info(f"Storing player prop for {name} - {bet_type} in Player Props Sheet")
                        # Figure out which team the player is on (approximate)
                        player_team = home_team
                        if name.lower() in away_team.lower() or away_team.lower() in name.lower():
                            player_team = away_team
                            
                        over_odds = ""
                        under_odds = ""
                        if "over" in bet_type.lower():
                            over_odds = odds
                        elif "under" in bet_type.lower():
                            under_odds = odds
                            
                        prop_data = {
                            "game_id": game.get("id", ""),
                            "player_name": name,
                            "team": player_team,
                            "prop_type": bet_type,
                            "line": str(line) if line else "",
                            "over_odds": str(over_odds) if over_odds else "",
                            "under_odds": str(under_odds) if under_odds else "",
                            "bookmaker": game.get("bookmakers", [{}])[0].get("title", "Unknown") if game.get("bookmakers") else "Unknown"
                        }
                        
                        # Store the prop in Google Sheets
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
            max_tokens=500, # Increased token limit for more detailed analysis
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
        
        confidence = rec_json.get('confidence', 75) # Default to 75% if not provided
        
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
                    
                    # Record metrics for prediction
                    try:
                        # Record prediction count metric
                        prediction_metrics = {
                            "type": "prediction_count",
                            "value": 1,
                            "sport": result.get("sport", "Unknown"),
                            "details": f"Straight bet: {result.get('recommendation', 'Unknown')}"
                        }
                        sheets_manager.update_metrics(prediction_metrics)
                        
                        # Record confidence metric
                        confidence_metrics = {
                            "type": "confidence_level",
                            "value": result.get("confidence", 0),
                            "sport": result.get("sport", "Unknown"),
                            "details": f"Straight bet: {result.get('recommendation', 'Unknown')}"
                        }
                        sheets_manager.update_metrics(confidence_metrics)
                    except Exception as e:
                        logger.error(f"Error updating metrics for prediction: {str(e)}")
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
            max_tokens=600, # Increased for more detailed analysis
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
        
        confidence = rec_json.get('confidence', 65) # Default to 65% for parlays
        
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
                    
                    # Record metrics for parlay prediction
                    try:
                        # Record prediction count metric
                        prediction_metrics = {
                            "type": "prediction_count",
                            "value": 1,
                            "sport": result.get("sport", "Unknown"),
                            "details": f"Parlay bet: {result.get('recommendation', 'Unknown')}"
                        }
                        sheets_manager.update_metrics(prediction_metrics)
                        
                        # Record confidence metric
                        confidence_metrics = {
                            "type": "confidence_level",
                            "value": result.get("confidence", 0),
                            "sport": result.get("sport", "Unknown"),
                            "details": f"Parlay bet: {result.get('recommendation', 'Unknown')}"
                        }
                        sheets_manager.update_metrics(confidence_metrics)
                    except Exception as e:
                        logger.error(f"Error updating metrics for parlay prediction: {str(e)}")
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
            max_tokens=500, # Increased for more detailed analysis
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
        
        confidence = rec_json.get('confidence', 70) # Default to 70% for player props
        
        # Check if 'player_bet' key exists - if not, extract from other keys or generate a recommendation
        player_bet = rec_json.get('player_bet')
        if not player_bet:
            logger.warning("No 'player_bet' key in AI response, attempting to construct recommendation from available data")
            # Try to construct from other keys
            if 'player' in rec_json and 'bet' in rec_json:
                player_bet = f"{rec_json['player']} on {rec_json['bet']}"
            elif 'recommendation' in rec_json:
                player_bet = rec_json['recommendation']
            elif 'bet' in rec_json:
                player_bet = rec_json['bet']
            else:
                # Fallback to a generic recommendation if all else fails
                player_bet = "Player not specified - insufficient data for clear recommendation"
        
        result = {
            "recommendation": player_bet,
            "explanation": rec_json.get('explanation', "No detailed explanation provided"),
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
                    
                    # Record metrics for player prop prediction
                    try:
                        # Record prediction count metric
                        prediction_metrics = {
                            "type": "prediction_count",
                            "value": 1,
                            "sport": result.get("sport", "Unknown"),
                            "details": f"Player prop bet: {result.get('recommendation', 'Unknown')}"
                        }
                        sheets_manager.update_metrics(prediction_metrics)
                        
                        # Record confidence metric
                        confidence_metrics = {
                            "type": "confidence_level",
                            "value": result.get("confidence", 0),
                            "sport": result.get("sport", "Unknown"),
                            "details": f"Player prop bet: {result.get('recommendation', 'Unknown')}"
                        }
                        sheets_manager.update_metrics(confidence_metrics)
                    except Exception as e:
                        logger.error(f"Error updating metrics for player prop prediction: {str(e)}")
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

def update_random_outcomes(limit: int = 5):
    """
    Update a random selection of pending outcomes for demo purposes.
    This would be replaced with actual outcome tracking in a production system.
    Args:
        limit: Maximum number of outcomes to update
    Returns:
        Count of updated outcomes
    """
    if not sheets_manager:
        return {"error": "Google Sheets integration not available"}
    
    try:
        # Get Predictions worksheet to find pending predictions
        predictions_sheet = sheets_manager.get_sheet("Predictions")
        if not predictions_sheet:
            return {"error": "Could not access Predictions worksheet"}
            
        # Get all rows from the Predictions sheet
        all_rows = predictions_sheet.get_all_values()
        if len(all_rows) <= 1:
            return {"message": "No predictions to update", "count": 0}
            
        # Extract header row and data rows
        header = all_rows[0]
        data_rows = all_rows[1:]
        
        # Find the column indexes for ID and outcome
        id_col = header.index("ID") if "ID" in header else 0
        outcome_col = header.index("Outcome") if "Outcome" in header else 7  # Default to 8th column
        
        # Find predictions with "Pending" outcome
        pending_predictions = []
        for i, row in enumerate(data_rows):
            if i < len(data_rows) and len(row) > outcome_col and row[outcome_col] == "Pending":
                pending_predictions.append({
                    "index": i + 2,  # +2 because of 0-based index and header row
                    "id": row[id_col],
                    "row": row
                })
        
        # If we don't have any pending predictions, return
        if not pending_predictions:
            return {"message": "No pending predictions to update", "count": 0}
            
        # Select predictions to update (limited by 'limit' parameter)
        to_update = random.sample(pending_predictions, min(limit, len(pending_predictions)))
        
        # Possible outcomes
        outcomes = ["Win", "Loss", "Push"]
        outcome_weights = [0.45, 0.45, 0.1]  # 45% win, 45% loss, 10% push
        
        updated_count = 0
        for pred in to_update:
            try:
                # Select a random outcome based on weights
                outcome = random.choices(outcomes, weights=outcome_weights, k=1)[0]
                
                # Update the Predictions sheet
                predictions_sheet.update_cell(pred["index"], outcome_col + 1, outcome)
                
                # Also record in Outcomes sheet
                outcome_data = {
                    "prediction_id": pred["id"],
                    "outcome": outcome,
                    "details": "Automated outcome update for demo",
                    "actual_result": f"Simulated {outcome.lower()} for {pred['row'][2]} bet"  # Sport from column 3
                }
                
                sheets_manager.store_outcome(outcome_data)
                
                # Also update metrics
                sheets_manager.update_metrics({
                    "type": "prediction_outcome",
                    "value": 1, 
                    "sport": pred["row"][2],  # Sport from column 3
                    "details": f"Outcome: {outcome} for {pred['row'][3]}"  # Recommendation from column 4
                })
                
                updated_count += 1
                logger.info(f"Updated outcome for prediction {pred['id']} to {outcome}")
            except Exception as e:
                logger.error(f"Error updating outcome for prediction {pred['id']}: {str(e)}")
        
        return {"message": f"Updated {updated_count} outcomes", "count": updated_count}
    except Exception as e:
        logger.error(f"Error in update_random_outcomes: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Error updating outcomes: {str(e)}"}

@app.get("/dashboard-metrics")
async def get_dashboard_metrics():
    """
    Get metrics for the dashboard.
    Returns:
        Dictionary with various metrics about predictions and outcomes
    """
    if not sheets_manager:
        return {
            "total_predictions": 0,
            "high_confidence_picks": 0,
            "average_confidence": 0,
            "sport_breakdown": []
        }
        
    try:
        # Get Predictions worksheet
        predictions_sheet = sheets_manager.get_sheet("Predictions")
        if not predictions_sheet:
            logger.error("Could not access Predictions worksheet")
            return {"error": "Could not access prediction data"}
            
        # Get all rows from Predictions sheet
        all_rows = predictions_sheet.get_all_values()
        if len(all_rows) <= 1:
            return {
                "total_predictions": 0,
                "high_confidence_picks": 0,
                "average_confidence": 0,
                "sport_breakdown": []
            }
            
        # Extract header and data
        header = all_rows[0]
        data_rows = all_rows[1:]
        
        # Find column indexes
        sport_col = header.index("Sport") if "Sport" in header else 2  # Default to 3rd column
        confidence_col = header.index("Confidence") if "Confidence" in header else 4  # Default to 5th column
        outcome_col = header.index("Outcome") if "Outcome" in header else 7  # Default to 8th column
        
        # Calculate metrics
        total_predictions = len(data_rows)
        high_confidence_count = 0
        confidence_sum = 0
        sport_counts = {}
        
        for row in data_rows:
            # Count high confidence predictions (>75%)
            try:
                confidence = float(row[confidence_col]) if row[confidence_col] and row[confidence_col] != "Confidence" else 0
                confidence_sum += confidence
                if confidence >= 75:
                    high_confidence_count += 1
            except (ValueError, IndexError):
                pass
                
            # Count by sport
            try:
                sport = row[sport_col] if len(row) > sport_col else "Unknown"
                if sport in sport_counts:
                    sport_counts[sport] += 1
                else:
                    sport_counts[sport] = 1
            except IndexError:
                pass
        
        # Calculate average confidence
        avg_confidence = round(confidence_sum / total_predictions, 1) if total_predictions > 0 else 0
        
        # Format sport breakdown
        sport_breakdown = [{"sport": sport, "count": count} for sport, count in sport_counts.items()]
        sport_breakdown.sort(key=lambda x: x["count"], reverse=True)
        
        # Get outcome summary
        outcomes_sheet = sheets_manager.get_sheet("Outcomes")
        win_count = 0
        loss_count = 0
        push_count = 0
        
        if outcomes_sheet:
            outcomes_rows = outcomes_sheet.get_all_values()
            if len(outcomes_rows) > 1:
                outcome_data_rows = outcomes_rows[1:]
                for row in outcome_data_rows:
                    if len(row) > 1:
                        outcome = row[1].lower() if row[1] else ""
                        if outcome == "win":
                            win_count += 1
                        elif outcome == "loss":
                            loss_count += 1
                        elif outcome == "push":
                            push_count += 1
        
        return {
            "total_predictions": total_predictions,
            "high_confidence_picks": high_confidence_count,
            "average_confidence": avg_confidence,
            "sport_breakdown": sport_breakdown[:5],  # Top 5 sports
            "outcome_summary": {
                "win": win_count,
                "loss": loss_count,
                "push": push_count,
                "win_rate": round((win_count / (win_count + loss_count)) * 100, 1) if (win_count + loss_count) > 0 else 0
            }
        }
    except Exception as e:
        logger.error(f"Error generating dashboard metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Error generating dashboard metrics: {str(e)}"}

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
        
        # Store each game in the Games tab
        if sheets_manager:
            logger.info(f"Storing {len(data)} games for {sp} in Games sheet")
            for game in data:
                game["sport"] = sp
                try:
                    sheets_manager.store_game(game)
                except Exception as e:
                    logger.error(f"Error storing game in Games sheet: {str(e)}")
        
        formatted_data = format_games_response(data)
        games_cache[cache_key] = formatted_data
        return formatted_data
    
    # If no sport specified, get all games
    all_games = []
    for sp, url in SPORTS_BASE_URLS.items():
        data = fetch_odds(API_KEY, url)
        if data:
            # Add sport to each game
            for game in data:
                game["sport"] = sp
                
                # Store each game in the Games tab
                if sheets_manager:
                    try:
                        sheets_manager.store_game(game)
                    except Exception as e:
                        logger.error(f"Error storing game in Games sheet: {str(e)}")
                        
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
            # Don't store games anymore, just process them for AI
            for game in data:
                game["sport"] = sp
            all_desc += format_odds_for_ai(data, sp)
            all_games.extend(data)
    
    result = generate_best_pick_with_ai(all_desc)
    bets_cache[cache_key] = result
    
    # Update metrics for new prediction
    if sheets_manager and result and not result.get("error"):
        try:
            # Add usage metrics
            metrics_data = {
                "type": "api_usage",
                "value": 1,
                "sport": "Overall",
                "details": "best_pick endpoint"
            }
            sheets_manager.update_metrics(metrics_data)
        except Exception as e:
            logger.error(f"Error updating metrics for API usage: {str(e)}")
    
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
            # Don't store games anymore, just process them for AI
            for game in data:
                game["sport"] = sp
            all_desc += format_odds_for_ai(data, sp)
    
    result = generate_best_parlay_with_ai(all_desc)
    bets_cache[cache_key] = result
    
    # Update metrics for API usage
    if sheets_manager and result and not result.get("error"):
        try:
            metrics_data = {
                "type": "api_usage",
                "value": 1,
                "sport": "Overall",
                "details": "best_parlay endpoint"
            }
            sheets_manager.update_metrics(metrics_data)
        except Exception as e:
            logger.error(f"Error updating metrics for API usage: {str(e)}")
    
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
    
    # Don't store games anymore, just format them for AI
    for game in data:
        game["sport"] = sp
    
    result = generate_best_pick_with_ai(format_odds_for_ai(data, sp))
    bets_cache[cache_key] = result
    
    # Update metrics for API usage
    if sheets_manager and result and not result.get("error"):
        try:
            metrics_data = {
                "type": "api_usage",
                "value": 1,
                "sport": sp,
                "details": "sport_best_pick endpoint"
            }
            sheets_manager.update_metrics(metrics_data)
        except Exception as e:
            logger.error(f"Error updating metrics for API usage: {str(e)}")
    
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
    
    # Don't store games anymore, just format them for AI
    for game in data:
        game["sport"] = sp
    
    result = generate_best_parlay_with_ai(format_odds_for_ai(data, sp))
    bets_cache[cache_key] = result
    
    # Update metrics for API usage
    if sheets_manager and result and not result.get("error"):
        try:
            metrics_data = {
                "type": "api_usage",
                "value": 1,
                "sport": sp,
                "details": "sport_best_parlay endpoint"
            }
            sheets_manager.update_metrics(metrics_data)
        except Exception as e:
            logger.error(f"Error updating metrics for API usage: {str(e)}")
    
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
    
    player_descriptions = []
    
    # 1) Try real player_props from Odds API
    try:
        odds_data = fetch_odds(API_KEY, base_url, markets="player_props")
        if odds_data:
            logger.info(f"Retrieved player props from odds API for {sport}")
            player_descriptions = format_player_odds_for_ai(odds_data, sp)
        else:
            logger.info(f"No player props from odds API for {sport}")
    except Exception as e:
        logger.error(f"Error fetching player props from odds API: {str(e)}")
        logger.error(traceback.format_exc())
    
    # 2) If none, fallback to TheSportsDB
    if not player_descriptions:
        logger.info(f"No player props from odds API for {sport}, trying TheSportsDB")
        try:
            thesports_data = fetch_player_data_thesportsdb(THESPORTSDB_API_KEY, sp)
            
            if thesports_data and isinstance(thesports_data, list):
                logger.info(f"Retrieved {len(thesports_data)} players from TheSportsDB for {sport}")
                
                # Process players and store in Player Props Sheet
                for player in thesports_data:
                    if not isinstance(player, dict):
                        continue
                        
                    name = player.get("strPlayer", "")
                    position = player.get("strPosition", "")
                    team = player.get("strTeam", "")
                    
                    if name and (position or team):
                        # Add to descriptions for AI
                        player_descriptions.append(f"{sp}: {name} - Position: {position}, Team: {team}")
                        
                        # Store in Player Props Sheet
                        if sheets_manager:
                            try:
                                logger.info(f"Storing player data for {name} from TheSportsDB")
                                prop_data = {
                                    "game_id": "",  # No game associated
                                    "player_name": name,
                                    "team": team,
                                    "prop_type": f"Position: {position}",
                                    "line": "",
                                    "over_odds": "",
                                    "under_odds": "",
                                    "bookmaker": "TheSportsDB"
                                }
                                sheets_manager.store_player_prop(prop_data)
                            except Exception as e:
                                logger.error(f"Error storing player data from TheSportsDB: {str(e)}")
            else:
                logger.warning(f"No valid player data returned from TheSportsDB for {sport}")
        except Exception as e:
            logger.error(f"Error processing TheSportsDB data: {str(e)}")
            logger.error(traceback.format_exc())
    
    # 3) If still none, report unavailable
    if not player_descriptions:
        logger.warning(f"No player data available for {sport} from any source")
        return {"best_player_bet": f"Player prop bets are unavailable for {sport}."}
    
    # Generate recommendation and store in Predictions sheet
    logger.info(f"Generating player bet recommendation from {len(player_descriptions)} player descriptions")
    result = generate_best_player_bet_with_ai(player_descriptions)
    bets_cache[cache_key] = result
    
    # Update metrics for API usage
    if sheets_manager and result and not result.get("error"):
        try:
            metrics_data = {
                "type": "api_usage",
                "value": 1,
                "sport": sp,
                "details": "player_best_bet endpoint"
            }
            sheets_manager.update_metrics(metrics_data)
        except Exception as e:
            logger.error(f"Error updating metrics for API usage: {str(e)}")
    
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

@app.post("/record-outcome")
async def record_outcome(outcome_data: OutcomeData):
    """
    Record the outcome of a prediction.
    Args:
        outcome_data: Outcome details including prediction ID and result
    Returns:
        Success message or error
    """
    if not sheets_manager:
        return {"error": "Google Sheets integration not available"}
    
    try:
        logger.info(f"Recording outcome for prediction {outcome_data.prediction_id}: {outcome_data.outcome}")
        # Convert Pydantic model to dictionary
        outcome_dict = outcome_data.dict()
        result = sheets_manager.store_outcome(outcome_dict)
        
        if result:
            # Record metrics for outcome
            try:
                metrics_data = {
                    "type": "prediction_outcome",
                    "value": 1,
                    "sport": "Unknown",  # We don't know the sport from just the outcome data
                    "details": f"Outcome: {outcome_data.outcome}"
                }
                sheets_manager.update_metrics(metrics_data)
            except Exception as e:
                logger.error(f"Error updating metrics for outcome: {str(e)}")
                
            return {"message": f"Outcome recorded successfully: {outcome_data.outcome}"}
        else:
            return {"error": "Failed to record outcome"}
    except Exception as e:
        logger.error(f"Error recording outcome: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Error recording outcome: {str(e)}"}

@app.post("/update-metrics")
async def update_metrics(
    metric_type: str = Query(..., description="Type of metric (e.g., predictions, confidence)"),
    value: float = Query(..., description="Metric value"),
    sport: str = Query("Overall", description="Sport code or 'Overall'"),
    details: str = Query(None, description="Additional details")
):
    """
    Update metrics in the Metrics sheet.
    Args:
        metric_type: Type of metric
        value: Metric value
        sport: Sport code
        details: Additional details
    Returns:
        Success message or error
    """
    if not sheets_manager:
        return {"error": "Google Sheets integration not available"}
    
    try:
        metrics_data = {
            "type": metric_type,
            "value": value,
            "sport": sport,
            "details": details or ""
        }
        
        success = sheets_manager.update_metrics(metrics_data)
        if success:
            return {"message": "Metrics updated successfully"}
        else:
            return {"error": "Failed to update metrics"}
    except Exception as e:
        logger.error(f"Error updating metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Error updating metrics: {str(e)}"}

@app.get("/update-demo-outcomes")
async def update_demo_outcomes(limit: int = Query(5, ge=1, le=20)):
    """
    Update a random selection of pending outcomes for demo purposes.
    Args:
        limit: Maximum number of outcomes to update
    Returns:
        Count of updated outcomes
    """
    return update_random_outcomes(limit)

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
            "/test-sheets-connection", # Endpoint for testing sheets connection
            "/check-service-account", # Endpoint to check service account details
            "/track-interaction",
            "/record-outcome", # New endpoint
            "/update-metrics", # New endpoint
            "/verify-sheets", # New debug endpoint
            "/test-all-sheets", # New test endpoint
            "/update-demo-outcomes", # New endpoint for demo outcomes
            "/dashboard-metrics" # New endpoint for dashboard data
        ]
    }

# Endpoint to check service account details
@app.get("/check-service-account")
async def check_service_account():
    """Check the service account email from credentials"""
    if not os.getenv("GOOGLE_CREDENTIALS_JSON"):
        return {"error": "No credentials JSON found in environment"}
    
    try:
        creds_json = json.loads(os.getenv("GOOGLE_CREDENTIALS_JSON"))
        service_account_email = creds_json.get("client_email", "Not found")
        project_id = creds_json.get("project_id", "Not found")
        return {
            "service_account_email": service_account_email,
            "project_id": project_id,
            "spreadsheet_id": os.getenv("SPREADSHEET_ID")
        }
    except Exception as e:
        return {"error": f"Failed to parse credentials: {str(e)}"}

# Endpoint for tracking user interactions
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
                # Record user interaction metric
                try:
                    metrics_data = {
                        "type": "user_interaction",
                        "value": 1,
                        "sport": "Unknown",  # We don't know the sport from the interaction data
                        "details": f"Interaction type: {interaction_type}, Page: {page}, Device: {device_type}"
                    }
                    sheets_manager.update_metrics(metrics_data)
                except Exception as e:
                    logger.error(f"Error updating metrics for user interaction: {str(e)}")
                    
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
        test_sheet = sheets_manager.get_sheet("Predictions")
        if test_sheet:
            row = ["TEST", "Connection Test", datetime.now().isoformat()]
            test_sheet.append_row(row)
            message = "Successfully connected to Google Sheets and wrote test data"
        else:
            message = "Connected to Google Sheets but couldn't access the Predictions worksheet"
            
        return {
            "status": "success",
            "message": message,
            "worksheets": worksheet_names
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error testing connection: {str(e)}",
            "error_details": traceback.format_exc()
        }

@app.get("/test-all-sheets")
async def test_all_sheets():
    """Test access to all Google Sheets tabs"""
    if not sheets_manager:
        return {"status": "error", "message": "Google Sheets integration not available"}
    
    sheet_names = ["Games", "Predictions", "Player Props Sheet", "Outcomes", "Metrics", "User Interactions Sheet"]
    results = {}
    
    for sheet_name in sheet_names:
        try:
            worksheet = sheets_manager.get_sheet(sheet_name)
            if worksheet:
                # Try to get the first row to verify read access
                header_row = worksheet.row_values(1)
                results[sheet_name] = {
                    "status": "success",
                    "message": f"Successfully accessed {sheet_name}",
                    "header": header_row
                }
            else:
                results[sheet_name] = {
                    "status": "error", 
                    "message": f"Could not access {sheet_name}"
                }
        except Exception as e:
            results[sheet_name] = {
                "status": "error",
                "message": f"Error accessing {sheet_name}: {str(e)}"
            }
    
    # Overall status
    all_success = all(result["status"] == "success" for result in results.values())
    
    return {
        "overall_status": "success" if all_success else "error",
        "sheets": results
    }

@app.get("/verify-sheets")
async def verify_sheets():
    """
    Debug endpoint to verify all Google Sheets tabs are working properly.
    Tests store operations on each tab.
    Returns:
        Results of tests on each sheet
    """
    if not sheets_manager:
        return {"status": "error", "message": "Google Sheets integration not available"}
    
    results = {}
    
    # 1. Test Games sheet
    try:
        logger.info("Testing Games sheet...")
        test_game = {
            "id": f"test_{uuid.uuid4()}",
            "sport": "TEST",
            "home_team": "Test Home",
            "away_team": "Test Away",
            "commence_time": datetime.now().isoformat(),
            "bookmakers": [
                {
                    "title": "Test Bookmaker",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Test Home", "price": 1.5},
                                {"name": "Test Away", "price": 2.5}
                            ]
                        }
                    ]
                }
            ],
            "status": "debug_test"
        }
        success = sheets_manager.store_game(test_game)
        results["Games"] = {
            "status": "success" if success else "error",
            "message": "Successfully stored test game" if success else "Failed to store test game"
        }
    except Exception as e:
        results["Games"] = {
            "status": "error",
            "message": f"Error testing Games sheet: {str(e)}"
        }
    
    # 2. Test Predictions sheet
    try:
        logger.info("Testing Predictions sheet...")
        test_prediction = {
            "type": "debug_test",
            "sport": "TEST",
            "recommendation": "Debug Test Prediction",
            "confidence": 50,
            "explanation": "This is a test prediction for debugging"
        }
        prediction_id = sheets_manager.store_prediction(test_prediction)
        results["Predictions"] = {
            "status": "success" if prediction_id else "error",
            "message": f"Successfully stored test prediction with ID: {prediction_id}" if prediction_id else "Failed to store test prediction",
            "prediction_id": prediction_id if prediction_id else None
        }
    except Exception as e:
        results["Predictions"] = {
            "status": "error",
            "message": f"Error testing Predictions sheet: {str(e)}"
        }
    
    # 3. Test Player Props sheet
    try:
        logger.info("Testing Player Props Sheet...")
        test_prop = {
            "game_id": f"test_{uuid.uuid4()}",
            "player_name": "Test Player",
            "team": "Test Team",
            "prop_type": "Test Prop",
            "line": "10.5",
            "over_odds": "1.90",
            "under_odds": "1.90",
            "bookmaker": "Test Bookmaker"
        }
        prop_id = sheets_manager.store_player_prop(test_prop)
        results["Player Props Sheet"] = {
            "status": "success" if prop_id else "error",
            "message": f"Successfully stored test player prop with ID: {prop_id}" if prop_id else "Failed to store test player prop"
        }
    except Exception as e:
        results["Player Props Sheet"] = {
            "status": "error",
            "message": f"Error testing Player Props Sheet: {str(e)}"
        }
    
    # 4. Test Outcomes sheet
    try:
        logger.info("Testing Outcomes sheet...")
        # Use the prediction_id from the Predictions test if available
        pred_id = results.get("Predictions", {}).get("prediction_id")
        if not pred_id:
            pred_id = f"test_{uuid.uuid4()}"
            
        test_outcome = {
            "prediction_id": pred_id,
            "outcome": "Test Outcome",
            "details": "Test details for debugging",
            "actual_result": "Test result"
        }
        success = sheets_manager.store_outcome(test_outcome)
        results["Outcomes"] = {
            "status": "success" if success else "error",
            "message": "Successfully stored test outcome" if success else "Failed to store test outcome"
        }
    except Exception as e:
        results["Outcomes"] = {
            "status": "error",
            "message": f"Error testing Outcomes sheet: {str(e)}"
        }
    
    # 5. Test Metrics sheet
    try:
        logger.info("Testing Metrics sheet...")
        test_metric = {
            "type": "debug_test",
            "value": 1,
            "sport": "TEST",
            "details": "Test metric for debugging"
        }
        success = sheets_manager.update_metrics(test_metric)
        results["Metrics"] = {
            "status": "success" if success else "error",
            "message": "Successfully stored test metric" if success else "Failed to store test metric"
        }
    except Exception as e:
        results["Metrics"] = {
            "status": "error",
            "message": f"Error testing Metrics sheet: {str(e)}"
        }
    
    # 6. Test User Interactions sheet
    try:
        logger.info("Testing User Interactions Sheet...")
        test_interaction = {
            "session_id": f"test_{uuid.uuid4()}",
            "prediction_id": results.get("Predictions", {}).get("prediction_id", f"test_{uuid.uuid4()}"),
            "interaction_type": "debug_test",
            "page": "test_page",
            "device_type": "test_device"
        }
        success = sheets_manager.store_user_interaction(test_interaction)
        results["User Interactions Sheet"] = {
            "status": "success" if success else "error",
            "message": "Successfully stored test interaction" if success else "Failed to store test interaction"
        }
    except Exception as e:
        results["User Interactions Sheet"] = {
            "status": "error",
            "message": f"Error testing User Interactions Sheet: {str(e)}"
        }
    
    # Overall status
    all_success = all(result["status"] == "success" for result in results.values())
    
    return {
        "overall_status": "success" if all_success else "error",
        "test_timestamp": datetime.now().isoformat(),
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run("rocketbetting:app", host="0.0.0.0", port=8000, reload=True)
