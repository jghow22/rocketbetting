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
from datetime import datetime, timezone, timedelta
import uuid
import traceback
from functools import lru_cache
import random
import math
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

# CORS settings - Enhanced for Wix compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www-redtapesoftwares-com.filesusr.com",
        "https://www-redtapesoftwares.com",
        "https://redtapesoftwares.wixsite.com",
        "https://www.redtapesoftwares.wixsite.com",
        "https://redtapesoftwares-com.filesusr.com",
        "https://www-redtapesoftwares-com.filesusr.com",
        "https://www.wix.com",
        "https://www.wixsite.com",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "*"  # Keep wildcard for debugging - will be removed in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*", "Origin", "X-Requested-With", "Content-Type", "Accept", "Authorization"],
    expose_headers=["*", "Access-Control-Allow-Origin", "Access-Control-Allow-Methods", "Access-Control-Allow-Headers"],
)

# Add CORS preflight handler
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle CORS preflight requests for all endpoints"""
    return {"message": "CORS preflight handled"}

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
    "NHL": "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds",
    "TENNIS": "https://api.the-odds-api.com/v4/sports/tennis_atp/odds"  # Added Tennis
}

# Define readable sport names
SPORT_DISPLAY_NAMES: Dict[str, str] = {
    "NBA": "Basketball (NBA)",
    "NFL": "Football (NFL)",
    "CFB": "College Football (NCAAF)",
    "MLS": "Soccer (MLS)",
    "MLB": "Baseball (MLB)",
    "NHL": "Hockey (NHL)",
    "TENNIS": "Tennis (ATP)"  # Added Tennis
}

# Define sport seasons (month ranges when sports are typically in season)
SPORT_SEASONS: Dict[str, Dict[str, int]] = {
    "NBA": {"start_month": 10, "end_month": 6},  # October to June
    "NFL": {"start_month": 9, "end_month": 2},   # September to February
    "CFB": {"start_month": 8, "end_month": 1},   # August to January
    "MLS": {"start_month": 2, "end_month": 12},  # February to December (year-round with breaks)
    "MLB": {"start_month": 3, "end_month": 11},  # March to November
    "NHL": {"start_month": 10, "end_month": 6},  # October to June
    "TENNIS": {"start_month": 1, "end_month": 12}  # Year-round (various tournaments)
}

def get_in_season_sports() -> List[str]:
    """
    Determine which sports are currently in season based on the current month.
    
    Returns:
        List of sport codes that are currently in season
    """
    current_month = datetime.now().month
    in_season_sports = []
    
    for sport, season in SPORT_SEASONS.items():
        start_month = season["start_month"]
        end_month = season["end_month"]
        
        # Handle seasons that span across year boundary
        if start_month <= end_month:
            # Normal season (e.g., March to November)
            if start_month <= current_month <= end_month:
                in_season_sports.append(sport)
        else:
            # Season spans year boundary (e.g., October to June)
            if current_month >= start_month or current_month <= end_month:
                in_season_sports.append(sport)
    
    logger.info(f"Sports currently in season (month {current_month}): {in_season_sports}")
    return in_season_sports

def get_primary_in_season_sport() -> str:
    """
    Get the primary sport that should be featured based on current season and popularity.
    
    Returns:
        Sport code for the primary in-season sport
    """
    in_season_sports = get_in_season_sports()
    
    if not in_season_sports:
        # Fallback to tennis if no major sports are in season
        return "TENNIS"
    
    # Priority order for sports (when multiple are in season)
    sport_priority = ["NBA", "NFL", "MLB", "NHL", "CFB", "MLS", "TENNIS"]
    
    for sport in sport_priority:
        if sport in in_season_sports:
            logger.info(f"Selected primary sport: {sport}")
            return sport
    
    # If none of the priority sports are in season, return the first available
    return in_season_sports[0]

# Create caches (TTL in seconds)
games_cache = TTLCache(maxsize=100, ttl=600) # Cache games for 10 minutes
bets_cache = TTLCache(maxsize=100, ttl=900)  # Cache bet recommendations for 15 minutes to reduce stale picks

def filter_games_by_date(games_data: List[Dict[str, Any]], current_day_only: bool = True) -> List[Dict[str, Any]]:
    """
    Filter games to only include current day and very near future events.
    
    Args:
        games_data: List of game data from API
        current_day_only: If True, only include games from today. If False, include today and next 7 days max.
    
    Returns:
        Filtered list of games
    """
    if not games_data:
        return []
    
    current_time = datetime.now(timezone.utc)
    current_date = current_time.date()
    
    # For current day only, we want games that start within the next 72 hours from now (more flexible)
    if current_day_only:
        # Include games starting within the next 72 hours (covers today and next 2 days)
        max_future_time = current_time + timedelta(hours=72)
        cutoff_time = current_time - timedelta(hours=2)  # 2 hour buffer for games that just started
    else:
        # For future games, limit to next 14 days maximum to avoid far-future games
        max_future_time = current_time + timedelta(days=14)
        cutoff_time = current_time - timedelta(hours=2)  # 2 hour buffer for games that just started
    
    filtered_games = []
    
    for game in games_data:
        if not game.get("commence_time"):
            logger.warning(f"Game missing commence_time: {game.get('home_team', 'Unknown')} vs {game.get('away_team', 'Unknown')}")
            continue
        
        try:
            # Parse the game time
            game_time_str = game["commence_time"]
            
            # Handle different time formats
            if game_time_str.endswith('Z'):
                game_time_str = game_time_str[:-1] + '+00:00'
            elif '+' not in game_time_str and 'T' in game_time_str:
                # Assume UTC if no timezone specified
                game_time_str = game_time_str + '+00:00'
            
            game_time = datetime.fromisoformat(game_time_str)
            
            # Check if game is within our time window (not too far in the past or future)
            if cutoff_time <= game_time <= max_future_time:
                # Additional validation: ensure the game has valid team names
                home_team = game.get('home_team', '').strip()
                away_team = game.get('away_team', '').strip()
                
                if home_team and away_team and home_team != 'TBD' and away_team != 'TBD':
                    filtered_games.append(game)
                    time_until_game = game_time - current_time
                    hours_until_game = time_until_game.total_seconds() / 3600
                    logger.info(f"Including game: {home_team} vs {away_team} at {game_time} ({hours_until_game:.1f} hours from now)")
                else:
                    logger.warning(f"Excluding game with invalid teams: {home_team} vs {away_team}")
            else:
                time_until_game = game_time - current_time
                hours_until_game = time_until_game.total_seconds() / 3600
                if game_time < cutoff_time:
                    logger.info(f"Excluding past game: {game.get('home_team', 'Unknown')} vs {game.get('away_team', 'Unknown')} at {game_time} ({hours_until_game:.1f} hours ago)")
                else:
                    logger.info(f"Excluding far-future game: {game.get('home_team', 'Unknown')} vs {game.get('away_team', 'Unknown')} at {game_time} ({hours_until_game:.1f} hours from now)")
                
        except Exception as e:
            logger.error(f"Error parsing game time for {game.get('home_team', 'Unknown')} vs {game.get('away_team', 'Unknown')}: {str(e)}")
            # For safety, exclude games with unparseable times
            continue
    
    logger.info(f"Filtered {len(games_data)} games down to {len(filtered_games)} games within time window")
    return filtered_games

def generate_current_day_tennis_predictions(match_type="straight", count=3):
    """
    Get real tennis predictions from actual tennis data.
    
    Args:
        match_type: Type of prediction ("straight", "parlay", or "player_prop")
        count: Number of predictions to generate
    
    Returns:
        List of formatted tennis match predictions for today
    """
    logger.info(f"Fetching real tennis data for {match_type} predictions")
    
    try:
        # Get real tennis data from the Odds API
        tennis_url = SPORTS_BASE_URLS.get("TENNIS")
        if not tennis_url:
            logger.error("Tennis URL not configured in SPORTS_BASE_URLS")
            return []
        
        # Fetch real tennis odds data
        tennis_data = fetch_odds(API_KEY, tennis_url)
        if not tennis_data:
            logger.error("No real tennis data available from API")
            return []
        
        # Filter for current day matches
        current_tennis_data = filter_games_by_date(tennis_data, current_day_only=True)
        
        if not current_tennis_data:
            logger.warning("No current day tennis matches available")
            return []
        
        logger.info(f"Found {len(current_tennis_data)} real tennis matches for today")
        
        # Format the real tennis data
        formatted_predictions = []
        for game in current_tennis_data[:count]:  # Limit to requested count
            home_team = game.get('home_team', 'Unknown')
            away_team = game.get('away_team', 'Unknown')
            commence_time = game.get('commence_time', '')
            bookmakers = game.get('bookmakers', [])
            
            # Get odds from first available bookmaker
            odds_info = ""
            if bookmakers and len(bookmakers) > 0:
                markets = bookmakers[0].get('markets', [])
                if markets and len(markets) > 0:
                    outcomes = markets[0].get('outcomes', [])
                    if len(outcomes) >= 2:
                        home_odds = outcomes[0].get('price', 0)
                        away_odds = outcomes[1].get('price', 0)
                        odds_info = f" | Odds: {home_team}: {home_odds}, {away_team}: {away_odds}"
            
            # Format the prediction
            prediction = f"TENNIS: {home_team} vs {away_team} on {commence_time}{odds_info} | Source: Real Tennis Data | Context: Tennis match, today"
            formatted_predictions.append(prediction)
        
        logger.info(f"Generated {len(formatted_predictions)} real tennis predictions")
        return formatted_predictions
        
    except Exception as e:
        logger.error(f"Error fetching real tennis data: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def generate_tennis_predictions_with_openai(match_type="straight", count=3):
    """
    Get real tennis predictions from actual tennis data.
    This function now only uses real data, no AI generation.
    
    Args:
        match_type: Type of prediction ("straight", "parlay", or "player_prop")
        count: Number of predictions to generate
    
    Returns:
        List of formatted tennis match predictions
    """
    logger.info(f"Fetching real tennis data for {match_type} predictions")
    
    try:
        # Get real tennis data from the Odds API
        tennis_url = SPORTS_BASE_URLS.get("TENNIS")
        if not tennis_url:
            logger.error("Tennis URL not configured in SPORTS_BASE_URLS")
            return []
        
        # Fetch real tennis odds data
        tennis_data = fetch_odds(API_KEY, tennis_url)
        if not tennis_data:
            logger.error("No real tennis data available from API")
            return []
        
        # Filter for current and future matches (less restrictive than current day only)
        current_tennis_data = filter_games_by_date(tennis_data, current_day_only=False)
        
        if not current_tennis_data:
            logger.warning("No current or upcoming tennis matches available")
            return []
        
        logger.info(f"Found {len(current_tennis_data)} real tennis matches")
        
        # Format the real tennis data
        formatted_predictions = []
        for game in current_tennis_data[:count]:  # Limit to requested count
            home_team = game.get('home_team', 'Unknown')
            away_team = game.get('away_team', 'Unknown')
            commence_time = game.get('commence_time', '')
            bookmakers = game.get('bookmakers', [])
            
            # Get odds from first available bookmaker
            odds_info = ""
            if bookmakers and len(bookmakers) > 0:
                markets = bookmakers[0].get('markets', [])
                if markets and len(markets) > 0:
                    outcomes = markets[0].get('outcomes', [])
                    if len(outcomes) >= 2:
                        home_odds = outcomes[0].get('price', 0)
                        away_odds = outcomes[1].get('price', 0)
                        odds_info = f" | Odds: {home_team}: {home_odds}, {away_team}: {away_odds}"
            
            # Format the prediction
            prediction = f"TENNIS: {home_team} vs {away_team} on {commence_time}{odds_info} | Source: Real Tennis Data | Context: Tennis match"
            formatted_predictions.append(prediction)
        
        logger.info(f"Generated {len(formatted_predictions)} real tennis predictions")
        return formatted_predictions
        
    except Exception as e:
        logger.error(f"Error fetching real tennis data: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def generate_tennis_recommendation_with_openai(bet_type="straight"):
    """
    Get real tennis betting recommendation from actual tennis data.
    This function now only uses real data, no AI generation.
    
    Args:
        bet_type: Type of bet - "straight", "parlay", or "player_prop"
    
    Returns:
        Complete recommendation object
    """
    logger.info(f"Fetching real tennis data for {bet_type} recommendation")
    
    try:
        # Get real tennis data from the Odds API
        tennis_url = SPORTS_BASE_URLS.get("TENNIS")
        if not tennis_url:
            logger.error("Tennis URL not configured in SPORTS_BASE_URLS")
            return None
        
        # Fetch real tennis odds data
        tennis_data = fetch_odds(API_KEY, tennis_url)
        if not tennis_data:
            logger.error("No real tennis data available from API")
            return None
        
        # Filter for current and future matches
        current_tennis_data = filter_games_by_date(tennis_data, current_day_only=False)
        
        if not current_tennis_data:
            logger.warning("No current or upcoming tennis matches available")
            return None
        
        # Get the first available match
        match = current_tennis_data[0]
        home_team = match.get('home_team', 'Unknown')
        away_team = match.get('away_team', 'Unknown')
        commence_time = match.get('commence_time', '')
        bookmakers = match.get('bookmakers', [])
        
        # Get odds from first available bookmaker
        home_odds = 0
        away_odds = 0
        if bookmakers and len(bookmakers) > 0:
            markets = bookmakers[0].get('markets', [])
            if markets and len(markets) > 0:
                outcomes = markets[0].get('outcomes', [])
                if len(outcomes) >= 2:
                    home_odds = outcomes[0].get('price', 0)
                    away_odds = outcomes[1].get('price', 0)
        
        # Create recommendation based on bet type
        if bet_type == "straight":
            # Choose the player with better odds
            if home_odds > 0 and away_odds > 0:
                if home_odds < away_odds:
                    recommended_player = home_team
                    odds = home_odds
                else:
                    recommended_player = away_team
                    odds = away_odds
            else:
                recommended_player = home_team
                odds = home_odds if home_odds > 0 else 2.0
            
            recommendation = {
                "sport": "TENNIS",
                "bet": f"{recommended_player} to win vs {away_team if recommended_player == home_team else home_team}",
                "explanation": f"Based on real tennis data: {recommended_player} has odds of {odds} in this match scheduled for {commence_time}. This recommendation is based on actual odds data from sportsbooks.",
                "confidence": 75,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "data_source": "Real Tennis Data"
            }
        
        elif bet_type == "parlay":
            # For parlay, we need multiple matches
            if len(current_tennis_data) >= 2:
                match1 = current_tennis_data[0]
                match2 = current_tennis_data[1]
                
                player1 = match1.get('home_team', 'Unknown')
                player2 = match2.get('home_team', 'Unknown')
                
                recommendation = {
                    "sport": "TENNIS",
                    "parlay": f"{player1} & {player2}",
                    "explanation": f"Parlay based on real tennis data: {player1} and {player2} both have upcoming matches. This recommendation uses actual odds data from sportsbooks.",
                    "confidence": 65,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "data_source": "Real Tennis Data"
                }
            else:
                logger.warning("Not enough tennis matches available for parlay")
                return None
        
        elif bet_type == "player_prop":
            recommendation = {
                "sport": "TENNIS",
                "player_bet": f"{home_team} - Match Winner in {home_team} vs {away_team}",
                "explanation": f"Player prop based on real tennis data: {home_team} vs {away_team} scheduled for {commence_time}. This recommendation uses actual odds data from sportsbooks.",
                "confidence": 70,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "data_source": "Real Tennis Data"
            }
        
        logger.info(f"Successfully generated tennis {bet_type} recommendation from real data")
        return recommendation
        
    except Exception as e:
        logger.error(f"Error fetching real tennis data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

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
        elif sport == "TENNIS":
            additional_context = f" | Context: Tennis match{', today' if is_today else ''}"  # Added tennis context
        
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
                            
                            # For tennis, we'll call them Player 1 and Player 2 instead of Home/Away
                            if sport == "TENNIS":
                                game_descriptions.append(
                                    f"{sport}: {home_team} vs {away_team}{game_time} | "
                                    f"Odds: {home_team}: {home_odds} ({home_implied_prob}% implied), "
                                    f"{away_team}: {away_odds} ({away_implied_prob}% implied) | "
                                    f"Source: {bookmaker_name}{additional_context}"
                                )
                            else:
                                game_descriptions.append(
                                    f"{sport}: {home_team} (Home) vs {away_team} (Away){game_time} | "
                                    f"Odds: {home_team}: {home_odds} ({home_implied_prob}% implied), "
                                    f"{away_team}: {away_odds} ({away_implied_prob}% implied) | "
                                    f"Source: {bookmaker_name}{additional_context}"
                                )
    
    return game_descriptions

# New helper to time-filter schedule games

def filter_schedule_games_by_time(schedule_games: List[Dict[str, Any]], max_hours: int = 48) -> List[Dict[str, Any]]:
    """
    Filter schedule games to those starting within the next max_hours hours (future only).
    Accepts games with either 'raw_date' (from schedule) or 'commence_time'.
    """
    if not schedule_games:
        return []
    
    current_time = datetime.now(timezone.utc)
    filtered: List[Dict[str, Any]] = []
    
    for game in schedule_games:
        try:
            ts = game.get("raw_date") or game.get("commence_time", "")
            if not ts:
                continue
            if ts.endswith('Z'):
                ts = ts[:-1] + '+00:00'
            elif '+' not in ts and 'T' in ts:
                ts = ts + '+00:00'
            dt = datetime.fromisoformat(ts)
            hours_until = (dt - current_time).total_seconds() / 3600.0
            if 0 <= hours_until <= max_hours:
                filtered.append(game)
        except Exception:
            continue
    
    return filtered

def format_schedule_games_for_ai(schedule_games: List[Dict[str, Any]], sport: str) -> List[str]:
    """
    Format schedule games data for AI analysis when odds data is not available.
    Args:
        schedule_games: List of games from schedule endpoint
        sport: Sport code (e.g., NBA, NFL)
    Returns:
        List of formatted game descriptions for AI analysis
    """
    game_descriptions = []
    
    # Get today's date for context
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    for game in schedule_games:
        home_team = game.get("homeTeam") or game.get("home_team")
        away_team = game.get("awayTeam") or game.get("away_team")
        commence_time = game.get("raw_date") or game.get("commence_time", "")
        game_sport = game.get("sport", sport)
        
        if not home_team or not away_team:
            continue
        
        # Format game time and determine if it's today
        game_time = ""
        is_today = False
        if commence_time:
            try:
                # Handle different time formats
                if commence_time.endswith('Z'):
                    commence_time = commence_time[:-1] + '+00:00'
                elif '+' not in commence_time and 'T' in commence_time:
                    commence_time = commence_time + '+00:00'
                
                dt = datetime.fromisoformat(commence_time)
                game_time = f" on {dt.strftime('%Y-%m-%d at %H:%M UTC')}"
                is_today = dt.strftime("%Y-%m-%d") == today
            except ValueError:
                pass
        
        # Add additional context based on sport
        additional_context = ""
        if game_sport == "NBA":
            additional_context = f" | Context: NBA basketball game{', today' if is_today else ''}"
        elif game_sport == "NHL":
            additional_context = f" | Context: NHL hockey game{', today' if is_today else ''}"
        elif game_sport == "NFL":
            additional_context = f" | Context: NFL football game{', today' if is_today else ''}"
        elif game_sport == "MLB":
            additional_context = f" | Context: MLB baseball game{', today' if is_today else ''}"
        elif game_sport == "MLS":
            additional_context = f" | Context: MLS soccer match{', today' if is_today else ''}"
        elif game_sport == "CFB":
            additional_context = f" | Context: College football game{', today' if is_today else ''}"
        elif game_sport == "TENNIS":
            additional_context = f" | Context: Tennis match{', today' if is_today else ''}"
        
        # For schedule games without odds, create a basic description
        if game_sport == "TENNIS":
            game_descriptions.append(
                f"{game_sport}: {home_team} vs {away_team}{game_time} | "
                f"Match scheduled - odds not yet available | "
                f"Source: Schedule{additional_context}"
            )
        else:
            game_descriptions.append(
                f"{game_sport}: {home_team} (Home) vs {away_team} (Away){game_time} | "
                f"Game scheduled - odds not yet available | "
                f"Source: Schedule{additional_context}"
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
        
        # For tennis, format the matchup differently since it's player vs player
        if sport == "TENNIS":
            matchup = f"{home_team} vs {away_team}" if home_team and away_team else ""
        else:
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
                
                # Tennis-specific context
                if sport == "TENNIS":
                    player_descriptions.append(
                        f"{sport}: {name} - {bet_type}{line_str} in {matchup}{game_time} | "
                        f"Odds: {odds} ({implied_prob}% implied probability) | "
                        f"Match: {matchup}"
                    )
                else:
                    player_descriptions.append(
                        f"{sport}: {name} - {bet_type}{line_str} in {matchup}{game_time} | "
                        f"Odds: {odds} ({implied_prob}% implied probability) | "
                        f"Teams: {home_team} (Home), {away_team} (Away)"
                    )
                
                # Store player prop in Google Sheets
                if sheets_manager:
                    try:
                        logger.info(f"Storing player prop for {name} - {bet_type} in Player Props Sheet")
                        
                        # For tennis, determine which player is which
                        player_team = ""
                        if sport == "TENNIS":
                            player_team = name  # In tennis, the player is their own "team"
                        else:
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

def standardize_betting_response(raw_response: Dict[str, Any], bet_type: str = "straight") -> Dict[str, Any]:
    """
    Standardize and normalize betting recommendation responses for consistency and clarity.
    Args:
        raw_response: Raw AI response dictionary
        bet_type: Type of bet (straight, parlay, player_prop)
    Returns:
        Standardized response dictionary
    """
    try:
        # Extract basic fields
        recommendation = raw_response.get("recommendation", "")
        explanation = raw_response.get("explanation", "")
        confidence = raw_response.get("confidence", 75)
        sport = raw_response.get("sport", "Unknown")
        
        # Normalize and clean the recommendation text
        recommendation = standardize_text(recommendation)
        explanation = standardize_text(explanation)
        
        # Create standardized format based on bet type
        if bet_type == "straight":
            standardized_recommendation = f"BET: {recommendation}"
            confidence_label = get_confidence_label(confidence)
            risk_level = get_risk_level(confidence)
            
        elif bet_type == "parlay":
            standardized_recommendation = f"PARLAY: {recommendation}"
            confidence_label = get_confidence_label(confidence)
            risk_level = get_risk_level(confidence)
            
        elif bet_type == "player_prop":
            standardized_recommendation = f"PLAYER PROP: {recommendation}"
            confidence_label = get_confidence_label(confidence)
            risk_level = get_risk_level(confidence)
            
        else:
            standardized_recommendation = recommendation
            confidence_label = get_confidence_label(confidence)
            risk_level = get_risk_level(confidence)
        
        # Create standardized explanation
        standardized_explanation = create_standardized_explanation(explanation, confidence, sport)
        
        return {
            "recommendation": standardized_recommendation,
            "explanation": standardized_explanation,
            "confidence": confidence,
            "confidence_label": confidence_label,
            "risk_level": risk_level,
            "sport": sport,
            "bet_type": bet_type,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "formatted_time": datetime.now(timezone.utc).strftime("%B %d, %Y at %I:%M %p UTC")
        }
        
    except Exception as e:
        logger.error(f"Error standardizing betting response: {str(e)}")
        return raw_response

def standardize_text(text: str) -> str:
    """
    Normalize and standardize text using text normalization techniques.
    Args:
        text: Raw text to standardize
    Returns:
        Standardized text
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove extra whitespace and normalize spacing
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Standardize common betting terms (keep them in proper case)
    text = re.sub(r'\b(win|wins|winning)\b', 'win', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(lose|loses|losing)\b', 'lose', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(beat|beats|beating)\b', 'beat', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(cover|covers|covering)\b', 'cover', text, flags=re.IGNORECASE)
    
    # Standardize confidence indicators (keep them in proper case)
    text = re.sub(r'\b(high|very high)\s+confidence\b', 'high confidence', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(medium|moderate)\s+confidence\b', 'medium confidence', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(low)\s+confidence\b', 'low confidence', text, flags=re.IGNORECASE)
    
    # Standardize odds references (keep them in proper case)
    text = re.sub(r'\b(odds|line)\b', 'odds', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(value|valuable)\b', 'value', text, flags=re.IGNORECASE)
    
    # Capitalize first letter of sentences
    text = '. '.join(sentence.capitalize() for sentence in text.split('. '))
    
    return text

def validate_recommendation_against_games(recommendation: Dict[str, Any], game_descriptions: List[str]) -> bool:
    """
    Validate that an AI recommendation actually matches a real game from the provided descriptions.
    
    Args:
        recommendation: The AI recommendation to validate
        game_descriptions: List of real game descriptions used to generate the recommendation
    
    Returns:
        True if the recommendation matches a real game, False otherwise
    """
    if not recommendation or not game_descriptions:
        return False
    
    recommended_team = recommendation.get('recommendation', '').strip()
    if not recommended_team:
        return False
    
    # Check if the recommended team appears in any of the real game descriptions
    for game_desc in game_descriptions:
        if recommended_team.lower() in game_desc.lower():
            logger.info(f"✅ Validated recommendation '{recommended_team}' against real game: {game_desc[:100]}...")
            return True
    
    logger.warning(f"❌ Recommendation '{recommended_team}' not found in any real games")
    return False

def get_confidence_label(confidence: int) -> str:
    """
    Convert confidence score to standardized label.
    Args:
        confidence: Confidence score (0-100)
    Returns:
        Standardized confidence label
    """
    if confidence >= 85:
        return "Very High"
    elif confidence >= 75:
        return "High"
    elif confidence >= 65:
        return "Medium"
    elif confidence >= 55:
        return "Low"
    else:
        return "Very Low"

def get_risk_level(confidence: int) -> str:
    """
    Convert confidence score to risk level.
    Args:
        confidence: Confidence score (0-100)
    Returns:
        Risk level string
    """
    if confidence >= 85:
        return "Low Risk"
    elif confidence >= 75:
        return "Medium-Low Risk"
    elif confidence >= 65:
        return "Medium Risk"
    elif confidence >= 55:
        return "Medium-High Risk"
    else:
        return "High Risk"

def create_standardized_explanation(explanation: str, confidence: int, sport: str) -> str:
    """
    Create a standardized explanation format.
    Args:
        explanation: Raw explanation text
        confidence: Confidence score
        sport: Sport name
    Returns:
        Standardized explanation
    """
    # Clean and standardize the explanation
    clean_explanation = standardize_text(explanation)
    
    # Add confidence context
    confidence_context = f"Confidence: {get_confidence_label(confidence)} ({confidence}%)"
    
    # Combine into standardized format (removed risk level since it's already in the header)
    standardized_explanation = f"{confidence_context}\n\nAnalysis:\n{clean_explanation}"
    
    return standardized_explanation

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
    
    # Add tennis-specific analysis guidance
    sport_specific_guidance = ""
    if sport_hint == "TENNIS":
        sport_specific_guidance = (
            "\n\nFor tennis specifically, focus on these critical factors:"
            "\n1. Player's recent form in the last 3-5 tournaments"
            "\n2. Head-to-head record between the players"
            "\n3. Surface compatibility (clay, grass, hard court specialists)"
            "\n4. Tournament stage and player's historical performance at this stage"
            "\n5. Physical condition and rest days between matches"
            "\n6. Playing style matchups (e.g., baseline vs. serve-and-volley)"
            "\n\nCRITICAL: Only recommend bets on matches that are confirmed to be scheduled in the future, not past matches."
            "\n\nVERY IMPORTANT: Your recommendation MUST explicitly state that it is for an upcoming match and include an approximate date."
        )
    
    # Add general date validation guidance for all sports
    sport_specific_guidance += (
        "\n\nCRITICAL DATE VALIDATION RULES:"
        "\n1. ONLY recommend bets on games/matches that are scheduled to happen in the future"
        "\n2. NEVER recommend bets on games that have already started or finished"
        "\n3. If you see a game time that's in the past, DO NOT recommend it"
        "\n4. Always verify the game date and time before making a recommendation"
        "\n5. If you're unsure about a game's timing, DO NOT recommend it"
        "\n6. Your recommendation MUST include the approximate date/time of the game"
    )
    
    # Limit the number of games to prevent token limit issues
    max_games = 25  # Limit to 25 games to stay within token limits
    limited_descriptions = game_descriptions[:max_games]
    
    if len(game_descriptions) > max_games:
        logger.info(f"Limited game descriptions from {len(game_descriptions)} to {max_games} to prevent token limit issues")
    
    # Log the games being sent to AI for debugging
    logger.info(f"Sending {len(limited_descriptions)} games to AI for analysis:")
    for i, desc in enumerate(limited_descriptions[:5]):  # Log first 5 games
        logger.info(f"  Game {i+1}: {desc[:100]}...")  # Log first 100 chars
    if len(limited_descriptions) > 5:
        logger.info(f"  ... and {len(limited_descriptions) - 5} more games")
    
    prompt = (
        "You are an expert sports betting analyst with deep knowledge of sports statistics, team dynamics, and betting strategy. "
        "Analyze the following games and recommend ONE specific bet that offers the best value, NOT simply the best odds. "
        "\n\nCRITICAL CONSTRAINT: You can ONLY recommend bets on teams that are actually playing in the games listed below. "
        "DO NOT create matchups between teams that are not scheduled to play each other. "
        "For example, if you see 'Seattle Mariners vs Chicago White Sox' and 'Pittsburgh Pirates vs Cincinnati Reds', "
        "you CANNOT recommend 'Seattle Mariners vs Pittsburgh Pirates' because they are not playing each other. "
        "You must pick ONE team from ONE of the actual games listed."
        "\n\nIn your analysis, consider the following factors, in order of importance:"
        "\n1. Recent team performance and momentum (last 5-10 games)"
        "\n2. Head-to-head matchups between the teams this season"
        "\n3. Key player availability (injuries, rest days, etc.)"
        "\n4. Home/away performance disparities"
        "\n5. Situational advantages (back-to-back games, travel fatigue, etc.)"
        "\n6. Statistical matchups and advantages"
        "\n7. Value compared to the offered odds"
        + sport_specific_guidance +  # Add tennis guidance if relevant
        "\n\nReturn ONLY a valid JSON object with no additional commentary. The JSON must follow EXACTLY this format:"
        '\n{"sport": "[Sport Name]", "bet": "[Team Name]", "explanation": "[Detailed reasoning with specific data points]", "confidence": [0-100], "game_time": "[Approximate date/time of the game]"}'
        "\n\nNote: Only assign confidence scores above 80 when you have extremely strong conviction backed by multiple data points."
        "\n\nCRITICAL: Your recommendation MUST include the game_time field with the approximate date/time of the upcoming game."
        "\n\nMake sure to mention EXPLICITLY in your explanation that this bet is for an UPCOMING match/game that will happen in the future."
        "\n\nAVAILABLE GAMES (you can ONLY recommend from these):"
        "\n" + sport_line + "\n" + "\n".join(limited_descriptions)
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
        logger.info(f"AI response for straight bet: {rec_text[:200]}...")  # Log first 200 chars
        
        # Try to parse as JSON
        try:
            rec_json = json.loads(rec_text)
        except json.JSONDecodeError:
            rec_json = extract_json(rec_text)
            if not rec_json:
                logger.error(f"JSON parsing error in straight bet response: {rec_text}")
                return {"error": f"Could not parse AI recommendation"}
        
        confidence = rec_json.get('confidence', 75) # Default to 75% if not provided
        
        # Create raw result
        raw_result = {
            "recommendation": f"{rec_json['bet']}",
            "explanation": rec_json['explanation'],
            "confidence": confidence,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "sport": rec_json.get('sport', sport_hint or "Unknown"),
            "game_time": rec_json.get('game_time', 'Unknown')
        }
        
        # Validate that the recommendation matches a real game
        if not validate_recommendation_against_games(raw_result, game_descriptions):
            logger.warning(f"AI recommendation doesn't match any real games: {raw_result.get('recommendation', 'Unknown')}")
            return {"error": "Unable to generate a valid recommendation for current games. Please try again later."}
        
        # Standardize the response
        result = standardize_betting_response(raw_result, "straight")
        
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
    
    # Add tennis-specific analysis guidance for parlays
    sport_specific_guidance = ""
    if sport_hint == "TENNIS":
        sport_specific_guidance = (
            "\n\nFor tennis specifically, focus on these critical factors when creating parlays:"
            "\n1. Player's recent form in the last 3-5 tournaments for all selections"
            "\n2. Head-to-head records between all players in your selections"
            "\n3. Surface compatibility for all players (clay, grass, hard court specialists)"
            "\n4. Tournament scheduling and potential fatigue factors"
            "\n5. Only include matches that are confirmed to be scheduled in the future"
            "\n6. Avoid combining highly volatile players in the same parlay"
            "\n\nVERY IMPORTANT: Your recommendation MUST explicitly state that these are upcoming matches and include approximate dates."
        )
    
    # Limit the number of games to prevent token limit issues
    max_games = 20  # Limit to 20 games to stay within token limits
    limited_descriptions = game_descriptions[:max_games]
    
    if len(game_descriptions) > max_games:
        logger.info(f"Limited game descriptions from {len(game_descriptions)} to {max_games} to prevent token limit issues")
    
    # Log the games being sent to AI for debugging
    logger.info(f"Sending {len(limited_descriptions)} games to AI for parlay analysis:")
    for i, desc in enumerate(limited_descriptions[:5]):  # Log first 5 games
        logger.info(f"  Game {i+1}: {desc[:100]}...")  # Log first 100 chars
    if len(limited_descriptions) > 5:
        logger.info(f"  ... and {len(limited_descriptions) - 5} more games")
    
    prompt = (
        "You are an expert sports betting analyst with deep knowledge of sports statistics, team dynamics, and betting strategy. "
        "Analyze the following games and create a 2-3 team parlay bet that offers the best value, NOT simply the highest potential payout. "
        "\n\nCRITICAL CONSTRAINT: You can ONLY recommend teams that are actually playing in the games listed below. "
        "DO NOT create matchups between teams that are not scheduled to play each other. "
        "For example, if you see 'Seattle Mariners vs Chicago White Sox' and 'Pittsburgh Pirates vs Cincinnati Reds', "
        "you CANNOT recommend 'Seattle Mariners & Pittsburgh Pirates' because they are not playing each other. "
        "You must pick teams from the actual games listed, and each team must be playing in their respective game."
        "\n\nIn your analysis, consider the following factors for EACH game in your parlay:"
        "\n1. Recent team performance and momentum (last 5-10 games)"
        "\n2. Head-to-head matchups between the teams this season"
        "\n3. Key player availability (injuries, rest days, etc.)"
        "\n4. Home/away performance disparities"
        "\n5. Situational advantages (back-to-back games, travel fatigue, etc.)"
        "\n6. Statistical matchups and advantages"
        "\n7. Diversification of risk (avoid multiple games with similar risk profiles)"
        + sport_specific_guidance +  # Add tennis guidance if relevant
        "\n\nReturn ONLY a valid JSON object with no additional commentary. The JSON must follow EXACTLY this format:"
        '\n{"sport": "[Sport Name]", "parlay": "[Team 1] & [Team 2] (add more teams if applicable)", "explanation": "[Detailed reasoning with specific data points for EACH pick]", "confidence": [0-100]}'
        "\n\nNote: Parlay confidence should generally be lower than straight bets due to compounding risk. Only assign confidence scores above 70 in extraordinary circumstances."
        "\n\nMake sure to mention EXPLICITLY in your explanation that this parlay is for UPCOMING matches/games that will happen in the future."
        "\n\nAVAILABLE GAMES (you can ONLY recommend teams from these):"
        "\n" + sport_line + "\n" + "\n".join(limited_descriptions)
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
        logger.info(f"AI response for parlay: {rec_text[:200]}...")  # Log first 200 chars
        
        # Try to parse as JSON
        try:
            rec_json = json.loads(rec_text)
        except json.JSONDecodeError:
            rec_json = extract_json(rec_text)
            if not rec_json:
                logger.error(f"JSON parsing error in parlay response: {rec_text}")
                return {"error": "Could not parse AI recommendation"}
        
        confidence = rec_json.get('confidence', 65) # Default to 65% for parlays
        
        # Create raw result
        raw_result = {
            "recommendation": f"{rec_json['parlay']}",
            "explanation": rec_json['explanation'],
            "confidence": confidence,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "sport": rec_json.get('sport', sport_hint or "Unknown")
        }
        
        # Validate the recommendation against actual games
        if not validate_parlay_recommendation_against_games(raw_result.get('recommendation', ''), game_descriptions):
            logger.warning(f"Parlay recommendation failed validation: {raw_result.get('recommendation', 'Unknown')}")
            return {"error": "Generated recommendation does not match available games. Please try again."}
        
        # Standardize the response
        result = standardize_betting_response(raw_result, "parlay")
        
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
    
    # Add tennis-specific analysis guidance for player props
    sport_specific_guidance = ""
    if sport_hint == "TENNIS":
        sport_specific_guidance = (
            "\n\nFor tennis specifically, consider these additional factors for player props:"
            "\n1. Player's historical performance on the current surface (clay, grass, hard court)"
            "\n2. Recent serving and return statistics"
            "\n3. Performance in similar tournament stages"
            "\n4. Fatigue from previous matches and tournament schedule"
            "\n5. Head-to-head history against similar playing styles"
            "\n6. Only recommend bets on matches that are confirmed to be scheduled in the future"
            "\n\nVERY IMPORTANT: Focus ONLY on upcoming matches in the next 1-7 days. DO NOT reference any matches that have already been played."
            "\nYour recommendation MUST explicitly mention that it is for an upcoming match and include an approximate date."
        )
    
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
        + sport_specific_guidance +  # Add tennis guidance if relevant
        "\n\nReturn ONLY a valid JSON object with no additional commentary. The JSON must follow EXACTLY this format:"
        '\n{"sport": "[Sport Name]", "player_bet": "[Player Name] on [Bet Type]", "explanation": "[Detailed reasoning with specific statistical evidence]", "confidence": [0-100]}'
        "\n\nYour explanation must include specific statistical data and clear reasoning."
        "\n\nMake sure to mention EXPLICITLY in your explanation that this bet is for an UPCOMING match/game that will happen in the future."
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
        
        # Create raw result
        raw_result = {
            "recommendation": player_bet,
            "explanation": rec_json.get('explanation', "No detailed explanation provided"),
            "confidence": confidence,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "sport": rec_json.get('sport', sport_hint or "Unknown")
        }
        
        # Standardize the response
        result = standardize_betting_response(raw_result, "player_prop")
        
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
            return {
                "total_predictions": 0,
                "high_confidence_picks": 0,
                "average_confidence": 0,
                "sport_breakdown": []
            }
        
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
        
        # Find column indexes with better error handling
        try:
            sport_col = header.index("Sport") if "Sport" in header else 2
            confidence_col = header.index("Confidence") if "Confidence" in header else 4
            outcome_col = header.index("Outcome") if "Outcome" in header else 7
        except (ValueError, IndexError):
            # Fallback to default columns if header parsing fails
            sport_col = 2
            confidence_col = 4
            outcome_col = 7
        
        # Calculate metrics with robust error handling
        total_predictions = len(data_rows)
        high_confidence_count = 0
        confidence_sum = 0
        valid_confidence_count = 0
        sport_counts = {}
        
        for row in data_rows:
            # Count high confidence predictions (>75%) with better error handling
            try:
                if len(row) > confidence_col and row[confidence_col]:
                    confidence_str = str(row[confidence_col]).strip()
                    if confidence_str and confidence_str.lower() not in ['confidence', 'n/a', 'nan', '']:
                        confidence = float(confidence_str)
                        if not math.isnan(confidence) and confidence >= 0:
                            confidence_sum += confidence
                            valid_confidence_count += 1
                            if confidence >= 75:
                                high_confidence_count += 1
            except (ValueError, IndexError, TypeError):
                pass
            
            # Count by sport with better error handling
            try:
                if len(row) > sport_col and row[sport_col]:
                    sport = str(row[sport_col]).strip()
                    if sport and sport.lower() not in ['sport', 'n/a', 'nan', '']:
                        if sport in sport_counts:
                            sport_counts[sport] += 1
                        else:
                            sport_counts[sport] = 1
            except (IndexError, TypeError):
                pass
        
        # Calculate average confidence with robust error handling
        if valid_confidence_count > 0:
            avg_confidence = round(confidence_sum / valid_confidence_count, 1)
        else:
            avg_confidence = 0
        
        # Ensure all values are valid numbers
        total_predictions = max(0, total_predictions)
        high_confidence_count = max(0, high_confidence_count)
        avg_confidence = max(0, min(100, avg_confidence))  # Clamp between 0-100
        
        # Format sport breakdown with validation
        sport_breakdown = []
        for sport, count in sport_counts.items():
            if count > 0 and isinstance(count, (int, float)) and not math.isnan(count):
                sport_breakdown.append({"sport": sport, "count": int(count)})
        sport_breakdown.sort(key=lambda x: x["count"], reverse=True)
        
        # Get outcome summary
        outcomes_sheet = sheets_manager.get_sheet("Outcomes")
        win_count = 0
        loss_count = 0
        push_count = 0
        
        if outcomes_sheet:
            try:
                outcomes_rows = outcomes_sheet.get_all_values()
                if len(outcomes_rows) > 1:
                    outcome_data_rows = outcomes_rows[1:]
                    for row in outcome_data_rows:
                        if len(row) > 1:
                            outcome = str(row[1]).lower().strip() if row[1] else ""
                            if outcome == "win":
                                win_count += 1
                            elif outcome == "loss":
                                loss_count += 1
                            elif outcome == "push":
                                push_count += 1
            except Exception as e:
                logger.warning(f"Error processing outcomes: {str(e)}")
        
        # Calculate win rate with error handling and validation
        total_outcomes = win_count + loss_count
        win_rate = round((win_count / total_outcomes) * 100, 1) if total_outcomes > 0 else 0
        
        # Ensure all outcome values are valid numbers
        win_count = max(0, win_count)
        loss_count = max(0, loss_count)
        push_count = max(0, push_count)
        win_rate = max(0, min(100, win_rate))  # Clamp between 0-100
        
        logger.info(f"Dashboard metrics calculated: predictions={total_predictions}, high_conf={high_confidence_count}, avg_conf={avg_confidence}")
        
        return {
            "total_predictions": total_predictions,
            "high_confidence_picks": high_confidence_count,
            "average_confidence": avg_confidence,
            "sport_breakdown": sport_breakdown[:5],  # Top 5 sports
            "outcome_summary": {
                "win": win_count,
                "loss": loss_count,
                "push": push_count,
                "win_rate": win_rate
            }
        }
    except Exception as e:
        logger.error(f"Error generating dashboard metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "total_predictions": 0,
            "high_confidence_picks": 0,
            "average_confidence": 0,
            "sport_breakdown": []
        }

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
        
        # For tennis, add special handling
        if sp == "TENNIS":
            logger.info("Fetching tennis matches with special handling")
            # Fetch from multiple tennis endpoints if needed (ATP, WTA, etc.)
            data = []
            tennis_endpoints = [
                "https://api.the-odds-api.com/v4/sports/tennis_atp/odds",
                "https://api.the-odds-api.com/v4/sports/tennis_wta/odds"
            ]
            
            for endpoint in tennis_endpoints:
                tennis_data = fetch_odds(API_KEY, endpoint)
                if tennis_data:
                    # Add sport to each game
                    for game in tennis_data:
                        game["sport"] = "TENNIS"
                    data.extend(tennis_data)
            
            # Filter for current and upcoming matches (less restrictive)
            data = filter_games_by_date(data, current_day_only=False)
            
            # If no current day real matches, use OpenAI to generate data
            if not data:
                logger.info("No real current day tennis matches found, generating with OpenAI")
                tennis_predictions = generate_current_day_tennis_predictions("straight", 5)
                
                # Transform these predictions into game data format
                if tennis_predictions:
                    for i, prediction in enumerate(tennis_predictions):
                        # Extract player names
                        match = re.search(r'TENNIS: (.*?) vs (.*?) on', prediction)
                        if match:
                            player1 = match.group(1).strip()
                            player2 = match.group(2).strip()
                            
                            # Extract date
                            date_match = re.search(r'on (\d{4}-\d{2}-\d{2})', prediction)
                            date_str = date_match.group(1) if date_match else None
                            
                            # If we have date, create a datetime
                            future_time = None
                            if date_str:
                                try:
                                    future_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                                    future_time = datetime.combine(future_date, datetime.min.time())
                                    future_time = future_time.replace(tzinfo=timezone.utc) + timedelta(hours=12)  # Noon UTC
                                except Exception:
                                    # Use a future date if parsing fails
                                    future_time = datetime.now(timezone.utc) + timedelta(days=i + 1, hours=12)
                            else:
                                # Use a future date if no date in the string
                                future_time = datetime.now(timezone.utc) + timedelta(days=i + 1, hours=12)
                            
                            # Extract odds if possible
                            odds1 = 2.0
                            odds2 = 2.0
                            odds_match = re.search(r'Odds: .*?: (\d+\.\d+).*?, .*?: (\d+\.\d+)', prediction)
                            if odds_match:
                                try:
                                    odds1 = float(odds_match.group(1))
                                    odds2 = float(odds_match.group(2))
                                except ValueError:
                                    pass
                            
                            # Create a game data structure
                            game_data = {
                                "id": f"openai_tennis_{uuid.uuid4()}",
                                "sport": "TENNIS",
                                "home_team": player1,
                                "away_team": player2,
                                "commence_time": future_time.isoformat(),
                                "bookmakers": [
                                    {
                                        "title": "Generated Odds",
                                        "markets": [
                                            {
                                                "key": "h2h",
                                                "outcomes": [
                                                    {"name": player1, "price": odds1},
                                                    {"name": player2, "price": odds2}
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                            data.append(game_data)
        else:
            url = SPORTS_BASE_URLS.get(sp)
            if not url:
                return {"error": f"Sport not supported: {sport}"}
            
            data = fetch_odds(API_KEY, url)
            if not data:
                return {"error": f"No games found for {sp}."}
        
        # Only store games that are actually displayed in the UI (limit to prevent rate limiting)
        if sheets_manager and len(data) <= 20:  # Only store if we have a reasonable number of games
            logger.info(f"Storing {len(data)} games for {sp} in Games sheet")
            for game in data:
                game["sport"] = sp
                try:
                    sheets_manager.store_game(game)
                except Exception as e:
                    logger.error(f"Error storing game in Games sheet: {str(e)}")
        elif sheets_manager and len(data) > 20:
            logger.info(f"Skipping storage of {len(data)} games for {sp} to prevent rate limiting")
        
        formatted_data = format_games_response(data)
        games_cache[cache_key] = formatted_data
        return formatted_data
    
    # If no sport specified, get all games
    all_games = []
    
    for sp, url in SPORTS_BASE_URLS.items():
        # Special handling for tennis
        if sp == "TENNIS":
            tennis_endpoints = [
                "https://api.the-odds-api.com/v4/sports/tennis_atp/odds",
                "https://api.the-odds-api.com/v4/sports/tennis_wta/odds"
            ]
            tennis_data = []
            
            for endpoint in tennis_endpoints:
                endpoint_data = fetch_odds(API_KEY, endpoint)
                if endpoint_data:
                    # Add sport to each game
                    for game in endpoint_data:
                        game["sport"] = "TENNIS"
                    tennis_data.extend(endpoint_data)
            
            # Filter for current and upcoming tennis matches (less restrictive)
            tennis_data = filter_games_by_date(tennis_data, current_day_only=False)
            
            # If no current day real matches, log the issue but don't generate fake data
            if not tennis_data:
                logger.info("No real current day tennis matches found for all sports view")
                logger.info("Tennis data will be excluded from recommendations until real matches are available")
            
            all_games.extend(tennis_data)
        else:
            data = fetch_odds(API_KEY, url)
            if data:
                # Add sport to each game
                for game in data:
                    game["sport"] = sp
                
                # Filter for current and upcoming matches (less restrictive)
                data = filter_games_by_date(data, current_day_only=False)
                
                # Only store games that are actually displayed in the UI (limit to prevent rate limiting)
                if sheets_manager and len(data) <= 20:  # Only store if we have a reasonable number of games
                    try:
                        for game in data:
                            sheets_manager.store_game(game)
                    except Exception as e:
                        logger.error(f"Error storing game in Games sheet: {str(e)}")
                elif sheets_manager and len(data) > 20:
                    logger.info(f"Skipping storage of {len(data)} games for {sp} to prevent rate limiting")
                
                all_games.extend(data)
    
    formatted_data = format_games_response(all_games)
    games_cache[cache_key] = formatted_data
    return formatted_data if formatted_data else {"error": "No games found."}

@app.get("/best-pick")
async def get_best_pick(
    fresh: bool = Query(False, description="Force fresh recommendation"),
    sport: str = Query(None, description="Optional sport filter")
):
    """
    Get the best straight bet recommendation across all sports.
    Args:
        fresh: If True, bypass cache and generate a new recommendation
        sport: Optional sport to filter by
    Returns:
        Dictionary with best pick recommendation
    """
    try:
        # Verify API keys (but don't throw errors)
        try:
            verify_api_keys()
        except Exception as e:
            logger.warning(f"API key verification failed: {str(e)}")
        
        cache_key = "best_pick:all"
        
        # Check if we have cached data and fresh parameter is not set
        if not fresh and cache_key in bets_cache:
            logger.info("Returning cached best pick recommendation")
            return {"best_pick": bets_cache[cache_key]}
        
        # Check if tennis is specifically requested
        tennis_requested = sport and sport.upper() == "TENNIS"
        
        if tennis_requested:
            # For tennis, use direct OpenAI recommendation
            logger.info("Tennis specifically requested, using OpenAI")
            direct_recommendation = generate_tennis_recommendation_with_openai("straight")
            if direct_recommendation:
                result = {
                    "recommendation": direct_recommendation.get("bet", ""),
                    "explanation": direct_recommendation.get("explanation", ""),
                    "confidence": direct_recommendation.get("confidence", 75),
                    "last_updated": direct_recommendation.get("last_updated", datetime.now(timezone.utc).isoformat()),
                    "sport": "TENNIS"
                }
                bets_cache[cache_key] = result
                return {"best_pick": result}
        
        all_desc = []
        
        # Get sports that are currently in season
        in_season_sports = get_in_season_sports()
        logger.info(f"Fetching data for in-season sports: {in_season_sports}")
        
        # Use the same data source as the schedule endpoint for consistency
        logger.info("Using schedule data source for betting recommendations")
        
        # Get games from the schedule endpoint
        try:
            schedule_response = await get_games()
            if isinstance(schedule_response, list) and schedule_response:
                logger.info(f"Found {len(schedule_response)} games in schedule")
                
                # Filter by requested sport if specified
                if sport:
                    requested_sport = sport.upper()
                    filtered_games = [game for game in schedule_response if game.get("sport", "").upper() == requested_sport]
                    logger.info(f"Filtered to {len(filtered_games)} games for {requested_sport}")
                else:
                    # Use all in-season sports
                    filtered_games = [game for game in schedule_response if game.get("sport", "").upper() in in_season_sports]
                    logger.info(f"Filtered to {len(filtered_games)} games for in-season sports")
                
                if filtered_games:
                    # Filter schedule games to next 48 hours, then take the next-up game, then format for AI
                    filtered_games = filter_schedule_games_by_time(filtered_games, max_hours=48)
                    filtered_games = sort_games_by_time(filtered_games)[:1]
                    all_desc = format_schedule_games_for_ai(filtered_games, sport or "ALL")
                    logger.info(f"Formatted {len(all_desc)} game descriptions for AI analysis from schedule (<=48h)")
                else:
                    logger.warning("No games found in schedule for betting recommendations")
            else:
                logger.warning("Schedule endpoint returned no data")
        except Exception as e:
            logger.error(f"Error getting games from schedule: {str(e)}")
            # Fallback to original method
            logger.info("Falling back to original odds API method")
            
            # Try to fetch real data first, but be more flexible with filtering
            for sp, url in SPORTS_BASE_URLS.items():
                # Skip sports that are not in season
                if sp not in in_season_sports:
                    logger.info(f"Skipping {sp} - not in season")
                    continue
                    
                try:
                    if sp == "TENNIS":
                        # Generate tennis data with OpenAI since real data is problematic
                        tennis_predictions = generate_current_day_tennis_predictions("straight", 3)
                        if tennis_predictions:
                            all_desc.extend(tennis_predictions)
                            logger.info(f"Added {len(tennis_predictions)} OpenAI current day tennis predictions")
                    else:
                        data = fetch_odds(API_KEY, url)
                        if data:
                            for game in data:
                                game["sport"] = sp
                            
                            # Try current day first, then expand to upcoming games if needed
                            filtered_data = filter_games_by_date(data, current_day_only=True)
                            if not filtered_data:
                                # If no current day games, try upcoming games (but still limited to 3 days)
                                filtered_data = filter_games_by_date(data, current_day_only=False)
                                logger.info(f"No current day games for {sp}, using upcoming games: {len(filtered_data)} found")
                            
                            if filtered_data:
                                # Only include games that are actually happening soon
                                soon_games = []
                                current_time = datetime.now(timezone.utc)
                                for game in filtered_data:
                                    try:
                                        game_time_str = game.get("commence_time", "")
                                        if game_time_str.endswith('Z'):
                                            game_time_str = game_time_str[:-1] + '+00:00'
                                        elif '+' not in game_time_str and 'T' in game_time_str:
                                            game_time_str = game_time_str + '+00:00'
                                        
                                        game_time = datetime.fromisoformat(game_time_str)
                                        time_until_game = game_time - current_time
                                        hours_until_game = time_until_game.total_seconds() / 3600
                                        
                                        # Only include games happening within the next 72 hours (3 days)
                                        if 0 <= hours_until_game <= 72:
                                            soon_games.append(game)
                                            logger.info(f"Adding soon game for {sp}: {game.get('home_team')} vs {game.get('away_team')} in {hours_until_game:.1f} hours")
                                    except Exception as e:
                                        logger.warning(f"Error processing game time for {sp}: {str(e)}")
                                        continue
                                
                                if soon_games:
                                    all_desc.extend(format_odds_for_ai(soon_games, sp))
                                    logger.info(f"Added {len(soon_games)} soon games for {sp}")
                                else:
                                    logger.warning(f"No soon games found for {sp}")
                            else:
                                logger.warning(f"No valid games found for {sp}")
                except Exception as e:
                    logger.warning(f"Error fetching odds for {sp}: {str(e)}")
        
        # Generate recommendation if we have data
        if all_desc:
            logger.info(f"Generating recommendation from {len(all_desc)} game descriptions")
            result = generate_best_pick_with_ai(all_desc)
            if result and not result.get("error"):
                bets_cache[cache_key] = result
                return {"best_pick": result}
            else:
                logger.warning(f"AI failed to generate valid recommendation: {result}")
        
        # If no real games available, provide a more helpful response
        logger.info("No real game data available, providing informative response")
        
        # Check what sports are actually available in the schedule
        try:
            schedule_response = await get_games()
            if isinstance(schedule_response, list) and schedule_response:
                available_sports = list(set([game.get("sport", "").upper() for game in schedule_response if game.get("sport")]))
                logger.info(f"Available sports in schedule: {available_sports}")
                
                if available_sports:
                    # Find the most relevant sport
                    if sport and sport.upper() in available_sports:
                        target_sport = sport.upper()
                    else:
                        target_sport = available_sports[0]
                    
                    # Get games for this sport
                    sport_games = [game for game in schedule_response if game.get("sport", "").upper() == target_sport]
                    
                    if sport_games:
                        # Filter schedule games to next 48 hours, then take the next-up game, then format for AI
                        sport_games = filter_schedule_games_by_time(sport_games, max_hours=48)
                        sport_games = sort_games_by_time(sport_games)[:1]
                        sport_desc = format_schedule_games_for_ai(sport_games, target_sport)
                        if sport_desc:
                            logger.info(f"Using {len(sport_desc)} games from schedule for {target_sport} (next-up <=48h)")
                            result = generate_best_pick_with_ai(sport_desc)
                            if result and not result.get("error"):
                                bets_cache[cache_key] = result
                                return {"best_pick": result}
                    
                    # If we have games but AI failed, provide a specific message
                    fallback_result = {
                        "recommendation": f"Check {target_sport} games in the schedule",
                        "explanation": f"Found {len(sport_games)} {target_sport} games in the schedule. Please review the game schedule for current betting opportunities.",
                        "confidence": 60,
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "sport": target_sport
                    }
                    bets_cache[cache_key] = fallback_result
                    return {"best_pick": fallback_result}
        except Exception as e:
            logger.error(f"Error checking schedule for fallback: {str(e)}")
        
        # If a specific sport was requested, provide sport-specific guidance
        if sport and sport.upper() in in_season_sports:
            requested_sport = sport.upper()
            logger.info(f"Generating sport-specific guidance for: {requested_sport}")
            
            fallback_result = {
                "recommendation": f"Review {requested_sport} schedule for current games",
                "explanation": f"Please check the {requested_sport} schedule for current games and betting opportunities. The system is currently updating game data.",
                "confidence": 50,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "sport": requested_sport
            }
            bets_cache[cache_key] = fallback_result
            return {"best_pick": fallback_result}
        
        # Final fallback with more helpful information
        return {"error": "No current betting opportunities available. Please check the game schedule for current matches and try again in a few minutes."}
            
        return {"error": "Unable to generate recommendation"}
        
    except Exception as e:
        logger.error(f"Unhandled error in get_best_pick: {str(e)}")
        return {"error": f"Failed to generate recommendation: {str(e)}"}

@app.get("/best-parlay")
async def get_best_parlay(
    fresh: bool = Query(False, description="Force fresh recommendation"),
    sport: str = Query(None, description="Optional sport filter")
):
    """
    Get the best parlay bet recommendation across all sports.
    Args:
        fresh: If True, bypass cache and generate a new recommendation
        sport: Optional sport to filter by
    Returns:
        Dictionary with best parlay recommendation
    """
    try:
        # Verify API keys (but don't throw errors)
        try:
            verify_api_keys()
        except Exception as e:
            logger.warning(f"API key verification failed: {str(e)}")
        
        cache_key = "best_parlay:all"
        
        # Check if we have cached data and fresh parameter is not set
        if not fresh and cache_key in bets_cache:
            logger.info("Returning cached best parlay recommendation")
            return {"best_parlay": bets_cache[cache_key]}
        
        # Check if tennis is specifically requested
        tennis_requested = sport and sport.upper() == "TENNIS"
        
        if tennis_requested:
            # For tennis, use direct OpenAI recommendation
            logger.info("Tennis specifically requested for parlay, using OpenAI")
            direct_recommendation = generate_tennis_recommendation_with_openai("parlay")
            if direct_recommendation:
                result = {
                    "recommendation": direct_recommendation.get("parlay", ""),
                    "explanation": direct_recommendation.get("explanation", ""),
                    "confidence": direct_recommendation.get("confidence", 65),
                    "last_updated": direct_recommendation.get("last_updated", datetime.now(timezone.utc).isoformat()),
                    "sport": "TENNIS"
                }
                bets_cache[cache_key] = result
                return {"best_parlay": result}
        
        all_desc = []
        
        # Get sports that are currently in season
        in_season_sports = get_in_season_sports()
        logger.info(f"Fetching data for in-season sports: {in_season_sports}")
        
        # Try to fetch real data first, but only for in-season sports
        for sp, url in SPORTS_BASE_URLS.items():
            # Skip sports that are not in season
            if sp not in in_season_sports:
                logger.info(f"Skipping {sp} - not in season")
                continue
            try:
                if sp == "TENNIS":
                    # Generate tennis data with OpenAI since real data is problematic
                    tennis_predictions = generate_current_day_tennis_predictions("parlay", 3)
                    if tennis_predictions:
                        all_desc.extend(tennis_predictions)
                        logger.info(f"Added {len(tennis_predictions)} OpenAI current day tennis predictions for parlay")
                else:
                    data = fetch_odds(API_KEY, url)
                    if data:
                        for game in data:
                            game["sport"] = sp
                        
                                                        # Filter for current day matches only, then apply additional time restrictions
                        data = filter_games_by_date(data, current_day_only=True)
                        
                        if data:
                            # Only include games that are actually happening soon
                            soon_games = []
                            current_time = datetime.now(timezone.utc)
                            for game in data:
                                try:
                                    game_time_str = game.get("commence_time", "")
                                    if game_time_str.endswith('Z'):
                                        game_time_str = game_time_str[:-1] + '+00:00'
                                    elif '+' not in game_time_str and 'T' in game_time_str:
                                        game_time_str = game_time_str + '+00:00'
                                    
                                    game_time = datetime.fromisoformat(game_time_str)
                                    time_until_game = game_time - current_time
                                    hours_until_game = time_until_game.total_seconds() / 3600
                                    
                                    # Only include games happening within the next 24 hours
                                    if 0 <= hours_until_game <= 24:
                                        soon_games.append(game)
                                        logger.info(f"Adding soon game for {sp}: {game.get('home_team')} vs {game.get('away_team')} in {hours_until_game:.1f} hours")
                                except Exception as e:
                                    logger.warning(f"Error processing game time for {sp}: {str(e)}")
                                    continue
                            
                            if soon_games:
                                # Check if this is schedule data or odds data
                                if soon_games and isinstance(soon_games, list) and len(soon_games) > 0:
                                    first_game = soon_games[0]
                                    if "homeTeam" in first_game or "awayTeam" in first_game:
                                        # This is schedule data
                                        all_desc.extend(format_schedule_games_for_ai(soon_games, sp))
                                        logger.info(f"Added {len(soon_games)} soon games for {sp} from schedule data")
                                    else:
                                        # This is odds data
                                        all_desc.extend(format_odds_for_ai(soon_games, sp))
                                        logger.info(f"Added {len(soon_games)} soon games for {sp} from odds data")
                                else:
                                    logger.warning(f"No valid games found for {sp}")
                            else:
                                logger.warning(f"No soon games found for {sp}")
            except Exception as e:
                logger.warning(f"Error fetching odds for {sp}: {str(e)}")
        
        # Generate recommendation if we have data
        if all_desc:
            result = generate_best_parlay_with_ai(all_desc)
            if result and not result.get("error"):
                bets_cache[cache_key] = result
                return {"best_parlay": result}
        
        # If no real games available, provide a more helpful response
        logger.info("No real game data available for parlay, providing informative response")
        
        # Check what sports are actually available in the schedule
        try:
            schedule_response = await get_games()
            if isinstance(schedule_response, list) and schedule_response:
                available_sports = list(set([game.get("sport", "").upper() for game in schedule_response if game.get("sport")]))
                logger.info(f"Available sports in schedule for parlay: {available_sports}")
                
                if available_sports:
                    # Find the most relevant sport
                    if sport and sport.upper() in available_sports:
                        target_sport = sport.upper()
                    else:
                        target_sport = available_sports[0]
                    
                    # Get games for this sport
                    sport_games = [game for game in schedule_response if game.get("sport", "").upper() == target_sport]
                    
                    if sport_games and len(sport_games) >= 2:  # Need at least 2 games for a parlay
                        # Keep only games in the next 48 hours and pick the next-up game
                        sport_games = filter_schedule_games_by_time(sport_games, max_hours=48)
                        data = sort_games_by_time(sport_games)[:1]
                        logger.info(f"Using {len(data)} next-up game(s) from schedule for {target_sport} (<=48h)")
                    else:
                        logger.warning(f"No games found in schedule for {target_sport}")
                        data = []
                    
                    if data:
                        # Format for AI
                        sport_desc = format_schedule_games_for_ai(data, target_sport)
                        if sport_desc:
                            logger.info(f"Using {len(sport_desc)} game descriptions for AI from schedule data")
                            result = generate_best_parlay_with_ai(sport_desc)
                            if result and not result.get("error"):
                                bets_cache[cache_key] = result
                                return {"best_parlay": result}
                    
                    # If we have games but AI failed, provide a specific message
                    fallback_result = {
                        "recommendation": f"Check {target_sport} games in the schedule for parlay opportunities",
                        "explanation": f"Found {len(sport_games)} {target_sport} games in the schedule. Please review the game schedule for current parlay betting opportunities.",
                        "confidence": 60,
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "sport": target_sport
                    }
                    bets_cache[cache_key] = fallback_result
                    return {"best_parlay": fallback_result}
        except Exception as e:
            logger.error(f"Error checking schedule for parlay fallback: {str(e)}")
        
        # Final fallback with more helpful information
        return {"error": "No current parlay opportunities available. Please check the game schedule for current matches and try again in a few minutes."}
        
    except Exception as e:
        logger.error(f"Unhandled error in get_best_parlay: {str(e)}")
        return {"error": f"Failed to generate recommendation: {str(e)}"}

@app.get("/sport-best-pick")
async def get_sport_best_pick(
    sport: str = Query(..., description="Sport code (e.g., NBA, NFL)"),
    fresh: bool = Query(False, description="Force fresh recommendation")
):
    """
    Get the best straight bet recommendation for a specific sport.
    Args:
        sport: Sport code to get recommendations for
        fresh: If True, bypass cache and generate a new recommendation
    Returns:
        Dictionary with best pick recommendation for the sport
    """
    # Verify API keys
    verify_api_keys()
    
    cache_key = f"best_pick:{sport}"
    
    # Check if we have cached data and fresh parameter is not set
    if not fresh and cache_key in bets_cache:
        logger.info(f"Returning cached best pick for {sport}")
        return {"sport_best_pick": bets_cache[cache_key]}
    
    sp = sport.upper()
    
    # Special handling for tennis - use real data only
    if sp == "TENNIS":
        logger.info("Fetching real tennis data for sport-specific best pick")
        
        # Get real tennis data from the Odds API
        tennis_url = SPORTS_BASE_URLS.get("TENNIS")
        if not tennis_url:
            return {"sport_best_pick": {"error": "Tennis URL not configured"}}
        
        # Fetch real tennis odds data
        tennis_data = fetch_odds(API_KEY, tennis_url)
        if not tennis_data:
            return {"sport_best_pick": {"error": "No real tennis data available from API"}}
        
        # Filter for current and future matches - CRITICAL: Only include future games
        current_tennis_data = filter_games_by_date(tennis_data, current_day_only=True)
        
        if not current_tennis_data:
            return {"sport_best_pick": {"error": "No current or upcoming tennis matches available"}}
        
        # Use the same logic as other sports - format data and generate AI recommendation
        for game in current_tennis_data:
            game["sport"] = sp
        
        result = generate_best_pick_with_ai(format_odds_for_ai(current_tennis_data, sp))
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
    else:
        # Handle non-tennis sports using schedule data for consistency
        logger.info(f"Using schedule data source for {sp} betting recommendations")
        
        try:
            # Get games from the schedule endpoint
            schedule_response = await get_games()
            if isinstance(schedule_response, list) and schedule_response:
                logger.info(f"Found {len(schedule_response)} games in schedule")
                
                # Filter for the specific sport
                filtered_games = [game for game in schedule_response if game.get("sport", "").upper() == sp]
                logger.info(f"Filtered to {len(filtered_games)} games for {sp}")
                
                if filtered_games:
                    # Keep only games in the next 48 hours
                    filtered_games = filter_schedule_games_by_time(filtered_games, max_hours=48)
                    data = filtered_games
                    logger.info(f"Using {len(filtered_games)} games from schedule for {sp} (<=48h)")
                else:
                    logger.warning(f"No games found in schedule for {sp}")
                    data = []
            else:
                logger.warning("Schedule endpoint returned no data")
                data = []
        except Exception as e:
            logger.error(f"Error getting games from schedule for {sp}: {str(e)}")
            # Fallback to original method
            logger.info(f"Falling back to original odds API method for {sp}")
            
            url = SPORTS_BASE_URLS.get(sp)
            if not url:
                return {"error": f"Sport not supported: {sport}"}
            
            data = fetch_odds(API_KEY, url)
            if not data:
                return {"error": f"No games found for {sp}."}
            
            # Add sport to each game
            for game in data:
                game["sport"] = sp
            
            # Filter for current day matches first, then expand if needed
            data = filter_games_by_date(data, current_day_only=True)
            logger.info(f"After current day filter for {sp}: {len(data)} games")
            
            if not data:
                # If no current day games, try upcoming games
                data = filter_games_by_date(data, current_day_only=False)
                logger.info(f"After expanding to upcoming games for {sp}: {len(data)} games")
            
            if data:
                # Only include games that are actually happening soon
                soon_games = []
                current_time = datetime.now(timezone.utc)
                for game in data:
                    try:
                        game_time_str = game.get("commence_time", "")
                        if game_time_str.endswith('Z'):
                            game_time_str = game_time_str[:-1] + '+00:00'
                        elif '+' not in game_time_str and 'T' in game_time_str:
                            game_time_str = game_time_str + '+00:00'
                        
                        game_time = datetime.fromisoformat(game_time_str)
                        time_until_game = game_time - current_time
                        hours_until_game = time_until_game.total_seconds() / 3600
                        
                        # Only include games happening within the next 48 hours (2 days)
                        if 0 <= hours_until_game <= 48:
                            soon_games.append(game)
                            logger.info(f"Adding soon game for {sp}: {game.get('home_team')} vs {game.get('away_team')} in {hours_until_game:.1f} hours")
                    except Exception as e:
                        logger.warning(f"Error processing game time for {sp}: {str(e)}")
                        continue
                
                if soon_games:
                    data = soon_games
                    logger.info(f"Using {len(soon_games)} soon games for {sp}")
                else:
                    logger.warning(f"No soon games found for {sp}")
                    data = []
        
        if not data:
            logger.warning(f"No games found for {sp}, providing informative response")
            
            # Provide a more helpful response when no games are available
            fallback_result = {
                "recommendation": f"No specific bet details available for {sp}",
                "explanation": f"No current or upcoming {sp} games found in the schedule. Please check the game schedule for current matches and try again in a few minutes.",
                "confidence": 0,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "sport": sp
            }
            bets_cache[cache_key] = fallback_result
            return {"sport_best_pick": fallback_result}
        
        # Debug logging
        logger.info(f"Generating best pick for {sp} with {len(data)} games")
        
        # Check if this is schedule data (has homeTeam/awayTeam) or odds data (has bookmakers)
        if data and isinstance(data, list) and len(data) > 0:
            first_game = data[0]
            if "homeTeam" in first_game or "awayTeam" in first_game:
                # This is schedule data
                formatted_data = format_schedule_games_for_ai(data, sp)
                logger.info(f"Formatted {len(formatted_data)} game descriptions for AI from schedule data")
            else:
                # This is odds data
                formatted_data = format_odds_for_ai(data, sp)
                logger.info(f"Formatted {len(formatted_data)} game descriptions for AI from odds data")
        else:
            formatted_data = []
            logger.warning("No data to format for AI analysis")
        
        result = generate_best_pick_with_ai(formatted_data)
        logger.info(f"Generated result for {sp}: {result}")
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
    sport: str = Query(..., description="Sport code (e.g., NBA, NFL)"),
    fresh: bool = Query(False, description="Force fresh recommendation")
):
    """
    Get the best parlay bet recommendation for a specific sport.
    Args:
        sport: Sport code to get recommendations for
        fresh: If True, bypass cache and generate a new recommendation
    Returns:
        Dictionary with best parlay recommendation for the sport
    """
    # Verify API keys
    verify_api_keys()
    
    cache_key = f"best_parlay:{sport}"
    
    # Check if we have cached data and fresh parameter is not set
    if not fresh and cache_key in bets_cache:
        logger.info(f"Returning cached best parlay for {sport}")
        return {"sport_best_parlay": bets_cache[cache_key]}
    
    sp = sport.upper()
    
    # Special handling for tennis
    if sp == "TENNIS":
        logger.info("Fetching tennis matches for sport-specific best parlay")
        # Use direct OpenAI recommendation for tennis
        direct_recommendation = generate_tennis_recommendation_with_openai("parlay")
        if direct_recommendation:
            result = {
                "recommendation": direct_recommendation.get("parlay", ""),
                "explanation": direct_recommendation.get("explanation", ""),
                "confidence": direct_recommendation.get("confidence", 65),
                "last_updated": direct_recommendation.get("last_updated", datetime.now(timezone.utc).isoformat()),
                "sport": "TENNIS"
            }
            bets_cache[cache_key] = result
            
            # Update metrics for API usage
            if sheets_manager:
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
        else:
            return {"sport_best_parlay": {"error": "Unable to generate tennis parlay recommendation"}}
    else:
        # Handle non-tennis sports using schedule data for consistency
        logger.info(f"Using schedule data source for {sp} parlay recommendations")
        
        try:
            # Get games from the schedule endpoint
            schedule_response = await get_games()
            if isinstance(schedule_response, list) and schedule_response:
                logger.info(f"Found {len(schedule_response)} games in schedule")
                
                # Filter for the specific sport
                filtered_games = [game for game in schedule_response if game.get("sport", "").upper() == sp]
                logger.info(f"Filtered to {len(filtered_games)} games for {sp}")
                
                if filtered_games:
                    # Keep only games in the next 48 hours
                    filtered_games = filter_schedule_games_by_time(filtered_games, max_hours=48)
                    data = filtered_games
                    logger.info(f"Using {len(filtered_games)} games from schedule for {sp} (<=48h)")
                else:
                    logger.warning(f"No games found in schedule for {sp}")
                    data = []
            else:
                logger.warning("Schedule endpoint returned no data")
                data = []
        except Exception as e:
            logger.error(f"Error getting games from schedule for {sp}: {str(e)}")
            # Fallback to original method
            logger.info(f"Falling back to original odds API method for {sp}")
            
            url = SPORTS_BASE_URLS.get(sp)
            if not url:
                return {"error": f"Sport not supported: {sport}"}
            
            data = fetch_odds(API_KEY, url)
            if not data:
                return {"error": f"No games found for {sp}."}
            
            # Add sport to each game
            for game in data:
                game["sport"] = sp
            
            # Filter for current and future matches (less restrictive)
            data = filter_games_by_date(data, current_day_only=False)
        
        if not data:
            logger.warning(f"No games found for {sp}, providing informative response")
            
            # Provide a more helpful response when no games are available
            fallback_result = {
                "recommendation": f"No specific bet details available for {sp}",
                "explanation": f"No current or upcoming {sp} games found in the schedule. Please check the game schedule for current matches and try again in a few minutes.",
                "confidence": 0,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "sport": sp
            }
            bets_cache[cache_key] = fallback_result
            return {"sport_best_parlay": fallback_result}
        
        # Debug logging
        logger.info(f"Generating best parlay for {sp} with {len(data)} games")
        
        # Check if this is schedule data (has homeTeam/awayTeam) or odds data (has bookmakers)
        if data and isinstance(data, list) and len(data) > 0:
            first_game = data[0]
            if "homeTeam" in first_game or "awayTeam" in first_game:
                # This is schedule data
                formatted_data = format_schedule_games_for_ai(data, sp)
                logger.info(f"Formatted {len(formatted_data)} game descriptions for AI from schedule data")
            else:
                # This is odds data
                formatted_data = format_odds_for_ai(data, sp)
                logger.info(f"Formatted {len(formatted_data)} game descriptions for AI from odds data")
        else:
            formatted_data = []
            logger.warning("No data to format for AI analysis")
        
        result = generate_best_parlay_with_ai(formatted_data)
        logger.info(f"Generated parlay result for {sp}: {result}")
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
    sport: str = Query(..., description="Sport code (e.g., NBA, NFL)"),
    fresh: bool = Query(False, description="Force fresh recommendation")
):
    """
    Get the best player prop bet recommendation for a specific sport.
    Args:
        sport: Sport code to get recommendations for
        fresh: If True, bypass cache and generate a new recommendation
    Returns:
        Dictionary with best player bet recommendation
    """
    # Verify API keys
    verify_api_keys()
    
    if sport.upper() == "OVERALL":
        return {"best_player_bet": "Please select a specific sport for player prop bets."}
    
    cache_key = f"best_player_bet:{sport}"
    
    # Check if we have cached data and fresh parameter is not set
    if not fresh and cache_key in bets_cache:
        logger.info(f"Returning cached best player bet for {sport}")
        return {"best_player_bet": bets_cache[cache_key]}
    
    sp = sport.upper()
    
    # Special handling for tennis - use real data only
    if sp == "TENNIS":
        logger.info("Fetching real tennis data for player prop bets")
        
        # Get real tennis data from the Odds API
        tennis_url = SPORTS_BASE_URLS.get("TENNIS")
        if not tennis_url:
            return {"best_player_bet": {"error": "Tennis URL not configured"}}
        
        # Fetch real tennis odds data
        tennis_data = fetch_odds(API_KEY, tennis_url)
        if not tennis_data:
            return {"best_player_bet": {"error": "No real tennis data available from API"}}
        
        # Filter for current day matches only, then apply additional time restrictions
        current_tennis_data = filter_games_by_date(tennis_data, current_day_only=True)
        
        if current_tennis_data:
            # Only include games that are actually happening soon
            soon_games = []
            current_time = datetime.now(timezone.utc)
            for game in current_tennis_data:
                try:
                    game_time_str = game.get("commence_time", "")
                    if game_time_str.endswith('Z'):
                        game_time_str = game_time_str[:-1] + '+00:00'
                    elif '+' not in game_time_str and 'T' in game_time_str:
                        game_time_str = game_time_str + '+00:00'
                    
                    game_time = datetime.fromisoformat(game_time_str)
                    time_until_game = game_time - current_time
                    hours_until_game = time_until_game.total_seconds() / 3600
                    
                    # Only include games happening within the next 48 hours (2 days)
                    if 0 <= hours_until_game <= 48:
                        soon_games.append(game)
                        logger.info(f"Adding soon tennis game: {game.get('home_team')} vs {game.get('away_team')} in {hours_until_game:.1f} hours")
                except Exception as e:
                    logger.warning(f"Error processing tennis game time: {str(e)}")
                    continue
            
            if soon_games:
                current_tennis_data = soon_games
                logger.info(f"Using {len(soon_games)} soon tennis games")
            else:
                logger.warning(f"No soon tennis games found")
                current_tennis_data = []
        
        if not current_tennis_data:
            return {"best_player_bet": {"error": "No current or upcoming tennis matches available"}}
        
        # Use the same logic as other sports - format data and generate AI recommendation
        for game in current_tennis_data:
            game["sport"] = sp
        
        result = generate_best_player_bet_with_ai(format_player_odds_for_ai(current_tennis_data, sp))
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
    
    base_url = SPORTS_BASE_URLS.get(sp)
    if not base_url:
        return {"best_player_bet": f"Sport not supported: {sport}"}
    
    player_descriptions = []
    success_source = None
    
    # 1) Try real player_props from Odds API
    try:
        logger.info(f"Attempting to fetch player props from Odds API for {sport}")
        odds_data = fetch_odds(API_KEY, base_url, markets="player_props")
        
        if odds_data:
            logger.info(f"Retrieved player props from odds API for {sport}: {len(odds_data)} games")
            
            # Filter for current day games only
            odds_data = filter_games_by_date(odds_data, current_day_only=True)
            
            player_descriptions = format_player_odds_for_ai(odds_data, sp)
            if player_descriptions:
                success_source = "Odds API"
                logger.info(f"Successfully formatted {len(player_descriptions)} player props from Odds API")
            else:
                logger.warning(f"No player descriptions could be formatted from Odds API data for {sport}")
        else:
            logger.info(f"No player props from odds API for {sport}")
    except Exception as e:
        logger.error(f"Error fetching player props from odds API: {str(e)}")
        logger.error(traceback.format_exc())
    
    # 3) If still none for MLB and MLS, use OpenAI
    if not player_descriptions and sp in ["MLB", "MLS"]:
        logger.warning(f"No player data available for {sp} from APIs, using OpenAI fallback")
        try:
            # Create a fallback prompt for OpenAI to generate plausible player props
            sport_full_name = "Baseball" if sp == "MLB" else "Soccer"
            league_name = "Major League Baseball" if sp == "MLB" else "Major League Soccer"
            
            fallback_prompt = f"""
            Generate 5 realistic player prop bets for upcoming {league_name} ({sp}) games.
            For each player, include:
            1. Player name (must be a real current {sp} player who is NOT injured)
            2. Their team
            3. The opponent team
            4. A realistic prop bet (e.g., hits, strikeouts for MLB; goals, assists for MLS)
            5. A realistic line for that prop
            6. Realistic odds
            
            CRITICAL: Do NOT include any injured players. For MLB, specifically exclude:
            - Jacob deGrom (injured)
            - Any player on the 60-day IL
            - Any player with recent injury reports
            
            Format each player prop as:
            "{sp}: [Player Name] - [Prop Type] [Line] in [Team] vs [Opponent Team] | Odds: [Odds] ([Implied Probability]% implied probability) | Teams: [Team] (Home), [Opponent] (Away)"
            
            Make these as realistic and accurate as possible for upcoming games.
            Include ONLY players who have games scheduled in the next few days AND are healthy/active.
            """
            
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=600,
                messages=[
                    {"role": "system", "content": f"You are an expert {sport_full_name} analyst who knows all current {sp} players and teams."},
                    {"role": "user", "content": fallback_prompt}
                ]
            )
            
            # Extract player props from OpenAI response
            fallback_text = response.choices[0].message.content.strip()
            player_descriptions = [line.strip() for line in fallback_text.split('\n') if sp in line and '-' in line]
            
            if player_descriptions:
                success_source = "OpenAI Fallback"
                logger.info(f"Generated {len(player_descriptions)} player props using OpenAI fallback for {sp}")
            else:
                logger.warning(f"Failed to generate player props using OpenAI fallback for {sp}")
        except Exception as e:
            logger.error(f"Error generating fallback player props with OpenAI: {str(e)}")
            logger.error(traceback.format_exc())
    
    # 4) If still no player data, report unavailable
    if not player_descriptions:
        logger.warning(f"No player data available for {sport} from any source")
        return {"best_player_bet": f"Player prop bets are unavailable for {sport}."}
    
    # Generate recommendation and store in Predictions sheet
    logger.info(f"Generating player bet recommendation from {len(player_descriptions)} player descriptions from {success_source}")
    result = generate_best_player_bet_with_ai(player_descriptions)
    
    # Add the data source to the result
    if isinstance(result, dict) and not result.get("error"):
        result["data_source"] = success_source
        bets_cache[cache_key] = result
        
        # Update metrics for API usage
        if sheets_manager and result and not result.get("error"):
            try:
                metrics_data = {
                    "type": "api_usage",
                    "value": 1,
                    "sport": sp,
                    "details": f"player_best_bet endpoint (source: {success_source})"
                }
                sheets_manager.update_metrics(metrics_data)
            except Exception as e:
                logger.error(f"Error updating metrics for API usage: {str(e)}")
        
        return {"best_player_bet": result}
    else:
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

@app.get("/in-season-sports")
async def get_in_season_sports_endpoint():
    """
    Get list of sports that are currently in season.
    Returns:
        Dictionary with in-season sports and primary sport
    """
    in_season_sports = get_in_season_sports()
    primary_sport = get_primary_in_season_sport()
    
    return {
        "in_season_sports": in_season_sports,
        "primary_sport": primary_sport,
        "display_names": {sport: SPORT_DISPLAY_NAMES.get(sport, sport) for sport in in_season_sports}
    }

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
            "/dashboard-metrics", # New endpoint for dashboard data
            "/test-prediction-api", # New test endpoint for API connectivity
            "/simple-pick" # New simple test endpoint
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

# Add new testing endpoints for troubleshooting
@app.get("/test-prediction-api")
async def test_prediction_api():
    """Test endpoint to check if AI prediction is working"""
    try:
        # Verify API keys are set
        api_keys_status = {
            "ODDS_API_KEY": bool(API_KEY),
            "OPENAI_API_KEY": bool(OPENAI_API_KEY),
            "THESPORTSDB_API_KEY": bool(THESPORTSDB_API_KEY)
        }
        
        # Test OpenAI connection
        openai_status = "Not tested"
        if OPENAI_API_KEY:
            try:
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    max_tokens=50,
                    messages=[
                        {"role": "system", "content": "You are a test assistant."},
                        {"role": "user", "content": "Return the text 'OpenAI API is working'"}
                    ]
                )
                openai_status = response.choices[0].message.content
            except Exception as e:
                openai_status = f"Error: {str(e)}"
        
        # Test odds API
        odds_status = "Not tested"
        if API_KEY:
            try:
                # Use NBA as a test sport
                test_url = SPORTS_BASE_URLS.get("NBA")
                if test_url:
                    data = fetch_odds(API_KEY, test_url)
                    odds_status = f"Success - found {len(data)} games" if data else "No games found"
                else:
                    odds_status = "No test URL found"
            except Exception as e:
                odds_status = f"Error: {str(e)}"
        
        # Test recommendation generation with minimal input
        ai_rec_status = "Not tested"
        try:
            test_desc = ["NBA: Test Home Team vs Test Away Team | Odds: 1.5, 2.5"]
            result = generate_best_pick_with_ai(test_desc)
            ai_rec_status = "Success" if result and not result.get("error") else f"Error: {result.get('error', 'Unknown error')}"
        except Exception as e:
            ai_rec_status = f"Error: {str(e)}"
        
        return {
            "api_keys": api_keys_status,
            "openai_test": openai_status,
            "odds_api_test": odds_status,
            "ai_recommendation_test": ai_rec_status
        }
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Test failed: {str(e)}"}

@app.get("/weather")
async def get_weather(
    lat: Optional[float] = Query(None, description="Latitude"),
    lon: Optional[float] = Query(None, description="Longitude"),
    location: Optional[str] = Query(None, description="Location name (city, country)")
):
    """
    Get weather forecast data from OpenWeatherMap API
    """
    try:
        # Get API key from environment
        weather_api_key = os.getenv('OPENWEATHER_API_KEY')
        logger.info(f"Weather API key found: {'Yes' if weather_api_key else 'No'}")
        
        if not weather_api_key:
            logger.error("Weather API key not configured in environment variables")
            raise HTTPException(status_code=500, detail="Weather API key not configured")
        
        # Build API URL based on parameters
        if lat is not None and lon is not None:
            # Use coordinates
            api_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={weather_api_key}&units=imperial"
            logger.info(f"Using coordinates: lat={lat}, lon={lon}")
        elif location:
            # Use location name - clean up the location format
            # Remove extra spaces and ensure proper format for OpenWeather
            clean_location = location.strip().replace('  ', ' ')
            api_url = f"https://api.openweathermap.org/data/2.5/forecast?q={clean_location}&appid={weather_api_key}&units=imperial"
            logger.info(f"Using location: {clean_location}")
        else:
            raise HTTPException(status_code=400, detail="Either lat/lon or location parameter is required")
        
        logger.info(f"Making request to OpenWeatherMap API: {api_url.replace(weather_api_key, '***')}")
        
        # Make request to OpenWeatherMap
        response = requests.get(api_url, timeout=10)
        
        # Log response details for debugging
        logger.info(f"OpenWeatherMap response status: {response.status_code}")
        logger.info(f"OpenWeatherMap response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            error_text = response.text
            logger.error(f"OpenWeatherMap API error: {response.status_code} - {error_text}")
            raise HTTPException(status_code=503, detail=f"Weather API error: {response.status_code} - {error_text}")
        
        weather_data = response.json()
        
        # Log successful request
        logger.info(f"Weather data retrieved successfully for {'coordinates' if lat and lon else 'location'}: {lat},{lon if lat and lon else location}")
        
        return weather_data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API request failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Weather service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting weather data: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/test-weather-key")
async def test_weather_key():
    """
    Test if the OpenWeather API key is configured and working
    """
    try:
        weather_api_key = os.getenv('OPENWEATHER_API_KEY')
        
        if not weather_api_key:
            return {
                "status": "error",
                "message": "Weather API key not configured",
                "details": "Please add OPENWEATHER_API_KEY to your environment variables"
            }
        
        # Test with a simple location - use a more reliable format
        test_url = f"https://api.openweathermap.org/data/2.5/forecast?q=London,UK&appid={weather_api_key}&units=imperial"
        
        logger.info(f"Testing weather API key with URL: {test_url.replace(weather_api_key, '***')}")
        
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            return {
                "status": "success",
                "message": "Weather API key is working",
                "api_key_length": len(weather_api_key),
                "test_location": "London"
            }
        elif response.status_code == 401:
            return {
                "status": "error",
                "message": "Weather API key is invalid or expired",
                "details": "Please check your OpenWeather API key"
            }
        else:
            return {
                "status": "error",
                "message": f"Weather API returned status {response.status_code}",
                "details": response.text
            }
            
    except Exception as e:
        logger.error(f"Error testing weather key: {str(e)}")
        return {
            "status": "error",
            "message": f"Error testing weather key: {str(e)}"
        }

@app.get("/simple-pick")
async def get_simple_pick():
    """
    Get a simple hardcoded pick for testing frontend integration.
    Returns:
        Dictionary with a test recommendation
    """
    test_pick = {
        "recommendation": "Test Team to win",
        "explanation": "This is a test recommendation to verify frontend-backend integration.",
        "confidence": 75,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "sport": "TEST"
    }
    
    return {"best_pick": test_pick}

@app.get("/test-cors")
async def test_cors():
    """
    Test endpoint specifically for CORS debugging.
    Returns:
        CORS test response with headers
    """
    return {
        "message": "CORS test endpoint working!",
        "timestamp": datetime.now().isoformat(),
        "origin": "test-cors-endpoint",
        "status": "success"
    }

def validate_parlay_recommendation_against_games(parlay_text: str, game_descriptions: List[str]) -> bool:
    """
    Validate a parlay recommendation by ensuring each referenced team appears
    in at least one of the provided game descriptions.
    
    Args:
        parlay_text: The parlay string (e.g., "Yankees & Braves")
        game_descriptions: List of formatted game descriptions sent to the AI
    Returns:
        True if at least two teams are identified and each appears in the games list
    """
    if not parlay_text or not game_descriptions:
        return False
    
    text = parlay_text.strip()
    # Strip common prefixes
    if text.upper().startswith("PARLAY:"):
        text = text.split(":", 1)[1].strip()
    
    # Split on common delimiters for parlays
    try:
        parts = re.split(r"\s*(?:&|\+|,|/| and )\s*", text, flags=re.IGNORECASE)
    except Exception:
        parts = [p.strip() for p in text.replace('&', ',').replace('+', ',').split(',')]
    
    # Normalize non-empty team tokens
    teams = [p.strip() for p in parts if p and p.strip()]
    if len(teams) < 2:
        logger.warning(f"Parlay validation found fewer than 2 team tokens in '{parlay_text}' -> {teams}")
        return False
    
    # Check each team appears in at least one game description
    games_text = [gd.lower() for gd in game_descriptions]
    for team in teams:
        team_lc = team.lower()
        if not any(team_lc in gd for gd in games_text):
            logger.warning(f"Parlay team '{team}' not found in any available games")
            return False
    
    logger.info(f"✅ Validated parlay teams against available games: {teams}")
    return True

# Helpers to parse and sort by game time

def parse_game_time_iso(game: Dict[str, Any]) -> Optional[datetime]:
    ts = game.get("raw_date") or game.get("commence_time", "")
    if not ts:
        return None
    try:
        if ts.endswith('Z'):
            ts = ts[:-1] + '+00:00'
        elif '+' not in ts and 'T' in ts:
            ts = ts + '+00:00'
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def sort_games_by_time(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        games,
        key=lambda g: (parse_game_time_iso(g) is None, parse_game_time_iso(g) or datetime.max.replace(tzinfo=timezone.utc))
    )

# ===================== NEW ENDPOINT: GAME-SPECIFIC BEST PICK =====================

# Cache individual game picks for a shorter period (5 min) because user might click through quickly.
game_pick_cache = TTLCache(maxsize=300, ttl=300)


@app.get("/game-best-pick")
async def get_game_best_pick(
    home_team: str = Query(..., description="Home team name exactly as in schedule"),
    away_team: str = Query(..., description="Away team name exactly as in schedule"),
    sport: str = Query(..., description="Sport code (e.g., MLB, NBA)"),
    fresh: bool = Query(False, description="Force regeneration even if cached")
):
    """Return a best straight-bet recommendation for the specific matchup that the user clicked in the schedule.

    We build a single-game description and feed it into the existing AI helper so the response format stays consistent.
    """

    try:
        cache_key = f"{sport}:{home_team}:{away_team}"
        if not fresh and cache_key in game_pick_cache:
            logger.info(f"Returning cached game pick for {cache_key}")
            return {"game_best_pick": game_pick_cache[cache_key]}

        sp = sport.upper()

        # Create a minimal game description to provide context to the AI.
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d at %H:%M UTC")
        description = f"{sp}: {home_team} (Home) vs {away_team} (Away) on {now_iso} | Source: UserSelectedSchedule"

        ai_result = generate_best_pick_with_ai([description])

        if ai_result.get("error"):
            return {"game_best_pick": ai_result}

        game_pick_cache[cache_key] = ai_result

        # Optional: log metrics to Sheets
        if sheets_manager and ai_result and not ai_result.get("error"):
            try:
                metrics_data = {
                    "type": "api_usage",
                    "value": 1,
                    "sport": sp,
                    "details": "game_best_pick endpoint"
                }
                sheets_manager.update_metrics(metrics_data)
            except Exception as e:
                logger.error(f"Error updating metrics for game_best_pick: {str(e)}")

        return {"game_best_pick": ai_result}
    except Exception as e:
        logger.error(f"Unhandled error in get_game_best_pick: {str(e)}")
        return {"error": f"Failed to generate game pick: {str(e)}"}

# Cache for individual game player props
game_player_cache = TTLCache(maxsize=300, ttl=300)


@app.get("/game-player-bet")
async def get_game_player_bet(
    home_team: str = Query(..., description="Home team name exactly as in schedule"),
    away_team: str = Query(..., description="Away team name exactly as in schedule"),
    sport: str = Query(..., description="Sport code (e.g., MLB, NBA)"),
    fresh: bool = Query(False, description="Force regeneration even if cached")
):
    """Return a player-prop recommendation for the specific matchup the user clicked."""

    try:
        cache_key = f"{sport}:{home_team}:{away_team}"
        if not fresh and cache_key in game_player_cache:
            logger.info(f"Returning cached game player bet for {cache_key}")
            return {"game_player_bet": game_player_cache[cache_key]}

        sp = sport.upper()

        # Build context for the AI – keep it simple so model focuses on these teams only
        description = [
            f"{sp}: Key players in upcoming matchup {home_team} vs {away_team}. Provide one high-confidence player prop bet for this game only."
        ]

        ai_result = generate_best_player_bet_with_ai(description)

        if ai_result.get("error"):
            return {"game_player_bet": ai_result}

        # Basic validation: make sure recommended player belongs to one of the teams
        rec_txt = json.dumps(ai_result).lower()
        if home_team.lower() not in rec_txt and away_team.lower() not in rec_txt:
            logger.warning("Player bet does not reference selected teams – flagging as error")
            return {"game_player_bet": {"error": "Model did not return player from selected game"}}

        game_player_cache[cache_key] = ai_result
        return {"game_player_bet": ai_result}

    except Exception as e:
        logger.error(f"Unhandled error in get_game_player_bet: {str(e)}")
        return {"error": f"Failed to generate game player bet: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("rocketbetting:app", host="0.0.0.0", port=8000, reload=True)
