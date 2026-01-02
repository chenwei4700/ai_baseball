"""
data_fetcher.py - 比賽資料抓取模組
負責抓取特定日期的比賽資料和球員歷史資料

來源: ai棒球專案
"""

from pybaseball import statcast, playerid_reverse_lookup
import pandas as pd
from datetime import datetime, timedelta


def get_game_data(game_date: str, team_code: str):
    """
    Fetches Statcast data for a specific date and team.
    
    Args:
        game_date (str): Date in 'YYYY-MM-DD' format.
        team_code (str): Team abbreviation (e.g., 'NYY', 'LAD').
        
    Returns:
        pd.DataFrame: Filtered DataFrame for the specific game, sorted by inning/play.
    """
    try:
        print(f"Fetching data for {game_date}...")
        df = statcast(start_dt=game_date, end_dt=game_date)
        
        if df is None or df.empty:
            return None
            
        # Filter by team (home or away)
        team_game_df = df[
            (df['home_team'] == team_code) | 
            (df['away_team'] == team_code)
        ].copy()
        
        if team_game_df.empty:
            return None
            
        # If multiple games (doubleheader), take the first game
        unique_games = team_game_df['game_pk'].unique()
        if len(unique_games) > 0:
            game_pk = unique_games[0]
            team_game_df = team_game_df[team_game_df['game_pk'] == game_pk]
        
        return team_game_df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def get_mlb_teams():
    """
    Returns a dictionary of MLB teams for the UI dropdown.
    """
    return {
        "NYY": "New York Yankees",
        "BOS": "Boston Red Sox",
        "LAD": "Los Angeles Dodgers",
        "SF": "San Francisco Giants",
        "TB": "Tampa Bay Rays",
        "TOR": "Toronto Blue Jays",
        "BAL": "Baltimore Orioles",
        "MIN": "Minnesota Twins",
        "CWS": "Chicago White Sox",
        "CLE": "Cleveland Guardians",
        "DET": "Detroit Tigers",
        "KC": "Kansas City Royals",
        "HOU": "Houston Astros",
        "SEA": "Seattle Mariners",
        "TEX": "Texas Rangers",
        "LAA": "Los Angeles Angels",
        "OAK": "Oakland Athletics",
        "ATL": "Atlanta Braves",
        "NYM": "New York Mets",
        "PHI": "Philadelphia Phillies",
        "MIA": "Miami Marlins",
        "WSH": "Washington Nationals",
        "MIL": "Milwaukee Brewers",
        "STL": "St. Louis Cardinals",
        "CHC": "Chicago Cubs",
        "PIT": "Pittsburgh Pirates",
        "CIN": "Cincinnati Reds",
        "ARI": "Arizona Diamondbacks",
        "COL": "Colorado Rockies",
        "SD": "San Diego Padres"
    }


def get_batters_from_game(df: pd.DataFrame):
    """
    Extracts unique batters from the game data.
    Returns a dict mapping batter_id to player_name.
    """
    if df is None or df.empty:
        return {}
    
    batter_ids = df['batter'].dropna().unique().tolist()
    
    if not batter_ids:
        return {}
    
    try:
        lookup_df = playerid_reverse_lookup(batter_ids, key_type='mlbam')
        
        if lookup_df.empty:
            return {}
        
        batters = {}
        for _, row in lookup_df.iterrows():
            batter_id = row['key_mlbam']
            name = f"{row['name_first']} {row['name_last']}"
            batters[batter_id] = name
            
        return batters
    except Exception as e:
        print(f"Error looking up batter names: {e}")
        return {}


def get_player_history(batter_id: int, game_date: str, days_back: int = 20):
    """
    Fetches historical Statcast data for a specific batter.
    
    Args:
        batter_id: MLBAM player ID.
        game_date: Reference date (YYYY-MM-DD).
        days_back: Number of days to look back.
        
    Returns:
        pd.DataFrame: All pitches faced by this batter in the lookback period.
    """
    try:
        end_date = datetime.strptime(game_date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=days_back)
        
        print(f"Fetching history for batter {batter_id} from {start_date.date()} to {end_date.date()}...")
        
        df = statcast(start_dt=start_date.strftime('%Y-%m-%d'), end_dt=end_date.strftime('%Y-%m-%d'))
        
        if df is None or df.empty:
            return None
        
        player_df = df[df['batter'] == batter_id].copy()
        
        return player_df if not player_df.empty else None
        
    except Exception as e:
        print(f"Error fetching player history: {e}")
        return None
