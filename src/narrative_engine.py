"""
narrative_engine.py - 比賽敘事生成引擎
負責提取比賽關鍵時刻並生成中英雙語戰報

來源: ai棒球專案
"""

import pandas as pd
import google.generativeai as genai
import re
import json


def extract_key_moments(df: pd.DataFrame, top_n=5):
    """
    Identifies the top N key moments from the game data based on Run Expectancy/Leverage.
    """
    action_plays = df[df['events'].notna() & (df['events'] != 'null')].copy()
    
    if action_plays.empty:
        return [], {}
    
    # Sort by absolute change in run expectancy (impact)
    if 'delta_run_exp' in action_plays.columns:
        action_plays['importance'] = action_plays['delta_run_exp'].abs()
    else:
        action_plays['importance'] = 0
    
    top_plays = action_plays.sort_values(by='importance', ascending=False).head(top_n)
    
    if 'at_bat_number' in top_plays.columns:
        top_plays = top_plays.sort_values('at_bat_number')
        
    moments = []
    for _, play in top_plays.iterrows():
        moment = {
            "game_date": play.get("game_date"),
            "inning": play.get("inning"),
            "inning_topbot": play.get("inning_topbot"),
            "outs": play.get("outs_when_up"),
            "home_score": play.get("home_score"),
            "away_score": play.get("away_score"),
            "batter": play.get("player_name") if 'player_name' in play else play.get("batter"), 
            "pitcher": play.get("pitcher"), 
            "event": play.get("events"),
            "description": play.get("des"),
            "metrics": {
                "release_speed": play.get("release_speed"),
                "pitch_type": play.get("pitch_type"),
                "launch_speed": play.get("launch_speed"),
                "launch_angle": play.get("launch_angle"),
                "hit_distance": play.get("hit_distance_sc")
            }
        }
        moments.append(moment)
        
    # Extract metadata
    final_home = df['home_score'].max()
    final_away = df['away_score'].max()
    
    home_team = df.iloc[0].get("home_team", "Home")
    away_team = df.iloc[0].get("away_team", "Away")
    
    if final_home > final_away:
        result = f"{home_team} wins"
    elif final_away > final_home:
        result = f"{away_team} wins"
    else:
        result = "Tie"
        
    metadata = {
        "home_team": home_team,
        "away_team": away_team,
        "home_score": final_home,
        "away_score": final_away,
        "result": result
    }
        
    return moments, metadata


def generate_game_narrative(moments, metadata, api_key):
    """
    Generates a narrative using Google Gemini API based on the extracted moments.
    Returns bilingual (English/Chinese) narrative.
    """
    if not moments:
        return "No key moments found for this game."
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        return f"Error configuring Gemini API: {e}"
    
    # System Prompt
    system_prompt = """
    You are a professional MLB sports writer. Your goal is to write a thrilling, dramatic game recap focusing on the key moments provided.
    
    Guidelines:
    1. **Storytelling**: Don't just list stats. Weave them into a narrative. Use dramatic language ("Rocketted", "Crushed", "Blazing").
    2. **Context**: 
       - Explicitly mention the teams playing: **{away_team} (Away) vs {home_team} (Home)**.
       - Mention user context clearly (e.g., "In the top of the 9th, with [Team] trailing...").
    3. **Metrics**: You MUST use the physical metrics provided (Exit Velocity, Distance, Pitch Speed) to enhance the description. 
    4. **Language**: 
       - First, write the narrative in **English**.
       - Second, write a **Traditional Chinese (繁體中文)** translation of the narrative. The Chinese version should be natural and use proper baseball terminology.
    5. **Conclusion**: You MUST end both versions by explicitly stating the **Final Score** and the **Winner**.
    6. **Format**: You MUST return the result in strictly valid JSON format. Structure:
    {{
        "english": "The English narrative...",
        "chinese": "The Chinese narrative..."
    }}
    """
    
    # User Prompt
    matchup_info = f"MATCHUP: {metadata.get('away_team')} (Visiting) @ {metadata.get('home_team')} (Home)\n"
    matchup_info += f"FINAL RESULT: {metadata.get('result')}. Score: {metadata.get('away_team')} {metadata.get('away_score')} - {metadata.get('home_team')} {metadata.get('home_score')}\n"
    
    user_prompt = f"{system_prompt.format(home_team=metadata.get('home_team'), away_team=metadata.get('away_team'))}\n\n{matchup_info}\nHere are the key moments from the game:\n\n"
    
    for i, m in enumerate(moments, 1):
        metrics = m['metrics']
        user_prompt += f"Moment {i}:\n"
        user_prompt += f"- Context: {m['inning_topbot']} of {m['inning']}, {m['outs']} outs. Score: Home {m['home_score']} - Away {m['away_score']}.\n"
        user_prompt += f"- Event: {m['event']}\n"
        user_prompt += f"- Description: {m['description']}\n"
        user_prompt += f"- Key Metrics: Pitch Speed {metrics.get('release_speed')}mph, Pitch {metrics.get('pitch_type')}, Exit Vel {metrics.get('launch_speed')}mph, Angle {metrics.get('launch_angle')}, Dist {metrics.get('hit_distance')}ft.\n\n"
        
    user_prompt += "Write the narrative in strictly valid JSON."
    
    try:
        response = model.generate_content(user_prompt)
        
        text = response.text
        
        # JSON extraction
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
            
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
            
        return json.loads(cleaned_text.strip())
            
    except Exception as e:
        return {"english": response.text if 'response' in locals() else str(e), "chinese": "Could not parse Chinese translation from AI response."}


def generate_player_analysis(game_df, history_df, player_name, api_key):
    """
    Generates an analysis of the pitching strategy used against a specific batter.
    Returns bilingual analysis.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        return {"english": f"Error configuring Gemini API: {e}", "chinese": ""}
    
    # Summarize Today's Game
    today_pitch_counts = game_df['pitch_type'].value_counts().to_dict() if 'pitch_type' in game_df.columns else {}
    today_results = game_df['events'].dropna().tolist() if 'events' in game_df.columns else []
    today_zones = game_df['zone'].value_counts().to_dict() if 'zone' in game_df.columns else {}
    
    # Summarize History
    if history_df is not None and not history_df.empty:
        hist_pitch_counts = history_df['pitch_type'].value_counts().to_dict() if 'pitch_type' in history_df.columns else {}
        hist_events = history_df[history_df['events'].notna()]['events'] if 'events' in history_df.columns else []
        hits = hist_events.isin(['single', 'double', 'triple', 'home_run']).sum()
        at_bats = (~hist_events.isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'catcher_interf'])).sum()
        recent_avg = round(hits / at_bats, 3) if at_bats > 0 else 0
        
        hist_hr = (hist_events == 'home_run').sum()
        hist_k = (hist_events == 'strikeout').sum()
    else:
        hist_pitch_counts = {}
        recent_avg = "N/A"
        hist_hr = 0
        hist_k = 0
    
    # Build Prompt
    system_prompt = """
    You are an expert MLB analyst. Analyze the pitching strategy used against the specified batter.
    
    Guidelines:
    1. **Today's Strategy**: Based on the pitch mix and zones today, describe the approach.
    2. **Historical Context**: Reference the batter's recent form (AVG, HR, K) to explain WHY the pitcher might have chosen this strategy.
    3. **Inference**: Conclude with a hypothesis on the pitcher's intent.
    4. **Language**: Provide analysis in English first, then Traditional Chinese.
    5. **Format**: Return JSON:
    {{
        "english": "English analysis...",
        "chinese": "Chinese analysis..."
    }}
    """
    
    user_prompt = f"""{system_prompt}

BATTER: {player_name}

TODAY'S PERFORMANCE:
- Pitches Seen: {today_pitch_counts}
- Zones Targeted: {today_zones}
- Outcomes: {today_results}

RECENT FORM (Last ~15 Games):
- Pitches Faced: {hist_pitch_counts}
- Batting Average: {recent_avg}
- Home Runs: {hist_hr}
- Strikeouts: {hist_k}

Analyze the strategy and provide the JSON output.
"""
    
    try:
        response = model.generate_content(user_prompt)
        
        text = response.text
        
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        return json.loads(text.strip())
        
    except Exception as e:
        return {"english": f"Error generating analysis: {e}", "chinese": ""}
