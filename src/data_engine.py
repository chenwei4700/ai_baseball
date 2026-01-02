"""
data_engine.py - 棒球資料引擎
負責抓取 MLB Statcast 資料、切片（Early/Mid/Late）與指標聚合計算

來源: baseball_ai_report專案

Module A: The Data Slicer (數據切片器)
Module B: The Metric Aggregator (指標聚合器)
"""

import pandas as pd
import numpy as np
from pybaseball import playerid_lookup, statcast_batter
from typing import Dict, Tuple, Any


def get_player_id(last_name: str, first_name: str) -> int:
    """
    查詢球員的 MLBAM ID
    
    Args:
        last_name: 球員姓氏 (e.g., "Ohtani")
        first_name: 球員名字 (e.g., "Shohei")
    
    Returns:
        int: 球員的 MLBAM ID
    
    Raises:
        ValueError: 找不到球員
    """
    lookup = playerid_lookup(last_name, first_name)
    
    if lookup.empty:
        raise ValueError(f"找不到球員: {first_name} {last_name}")
    
    player_id = lookup.iloc[0]['key_mlbam']
    return int(player_id)


def fetch_statcast_data(player_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    """
    抓取球員的 Statcast 逐球資料 (僅限例行賽)
    
    Args:
        player_id: MLBAM 球員 ID
        start_date: 開始日期 (YYYY-MM-DD)
        end_date: 結束日期 (YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: Statcast 資料 (僅例行賽)
    """
    df = statcast_batter(start_date, end_date, player_id)
    
    if df.empty:
        raise ValueError(f"該期間沒有資料: {start_date} ~ {end_date}")
    
    # 過濾僅保留例行賽 (Regular Season)
    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'R']
        if df.empty:
            raise ValueError(f"該期間沒有例行賽資料: {start_date} ~ {end_date}")
    
    return df


def slice_by_game_index(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    依據出賽序號切割資料為 Early/Mid/Late 三段
    
    Args:
        df: 原始 Statcast 資料
    
    Returns:
        Tuple: (early_df, mid_df, late_df) 各 10 場比賽的資料
    
    Raises:
        ValueError: 出賽場次不足 30 場
    """
    df = df.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    game_dates = df['game_date'].drop_duplicates().sort_values().reset_index(drop=True)
    n_games = len(game_dates)
    
    if n_games < 30:
        raise ValueError(f"樣本不足！該球員整季僅出賽 {n_games} 場，需至少 30 場。")
    
    game_index_map = {date: idx for idx, date in enumerate(game_dates)}
    df['game_index'] = df['game_date'].map(game_index_map)
    
    # Segment A (Early): 前 10 場
    early_dates = game_dates[0:10].tolist()
    early_df = df[df['game_date'].isin(early_dates)]
    
    # Segment B (Mid): 中間 10 場
    mid_center = n_games // 2
    mid_start = mid_center - 5
    mid_end = mid_center + 5
    mid_dates = game_dates[mid_start:mid_end].tolist()
    mid_df = df[df['game_date'].isin(mid_dates)]
    
    # Segment C (Late): 最後 10 場
    late_dates = game_dates[-10:].tolist()
    late_df = df[df['game_date'].isin(late_dates)]
    
    return early_df, mid_df, late_df


def aggregate_metrics(segment_df: pd.DataFrame, segment_name: str) -> Dict[str, Any]:
    """
    計算單一區段的 10 項關鍵指標
    
    Args:
        segment_df: 該區段的資料
        segment_name: 區段名稱 (Early/Mid/Late)
    
    Returns:
        Dict: 包含 10 項指標的字典
    """
    df = segment_df.copy()
    
    metrics = {
        'segment': segment_name,
        'games': df['game_date'].nunique(),
        'plate_appearances': len(df[df['events'].notna()]),
    }
    
    # 1. 物理爆發力：launch_speed 平均值
    valid_launch = df['launch_speed'].dropna()
    metrics['avg_launch_speed'] = round(valid_launch.mean(), 2) if len(valid_launch) > 0 else None
    
    # 2. 擊球技巧：launch_angle 平均值
    valid_angle = df['launch_angle'].dropna()
    metrics['avg_launch_angle'] = round(valid_angle.mean(), 2) if len(valid_angle) > 0 else None
    
    # 3. 體能穩定性：Hard Hit Rate (launch_speed > 95 的比例)
    if len(valid_launch) > 0:
        hard_hits = (valid_launch > 95).sum()
        metrics['hard_hit_rate'] = round(hard_hits / len(valid_launch) * 100, 2)
    else:
        metrics['hard_hit_rate'] = None
    
    # 4. 辨識能力：Whiff Rate (swinging_strike 比例)
    total_swings = df[df['description'].isin([
        'swinging_strike', 'swinging_strike_blocked', 
        'foul', 'foul_tip', 'hit_into_play'
    ])]
    swinging_strikes = df[df['description'].isin(['swinging_strike', 'swinging_strike_blocked'])]
    if len(total_swings) > 0:
        metrics['whiff_rate'] = round(len(swinging_strikes) / len(total_swings) * 100, 2)
    else:
        metrics['whiff_rate'] = None
    
    # 5. 巔峰力量：hit_distance_sc 最大值
    valid_distance = df['hit_distance_sc'].dropna()
    metrics['max_hit_distance'] = round(valid_distance.max(), 2) if len(valid_distance) > 0 else None
    
    # 6. 進壘威脅：release_spin_rate 平均值
    valid_spin = df['release_spin_rate'].dropna()
    metrics['avg_pitcher_spin'] = round(valid_spin.mean(), 2) if len(valid_spin) > 0 else None
    
    # 7-9. 事件統計
    events = df['events'].dropna()
    total_events = len(events)
    
    home_runs = (events == 'home_run').sum()
    metrics['home_runs'] = int(home_runs)
    
    walks = (events == 'walk').sum()
    metrics['walks'] = int(walks)
    metrics['bb_rate'] = round(walks / total_events * 100, 2) if total_events > 0 else None
    
    strikeouts = (events == 'strikeout').sum()
    metrics['strikeouts'] = int(strikeouts)
    metrics['k_rate'] = round(strikeouts / total_events * 100, 2) if total_events > 0 else None
    
    # 10. 運氣成分：BABIP
    hits = events.isin(['single', 'double', 'triple', 'home_run']).sum()
    at_bats = events.isin([
        'single', 'double', 'triple', 'home_run',
        'field_out', 'strikeout', 'double_play', 
        'grounded_into_double_play', 'force_out',
        'fielders_choice', 'fielders_choice_out'
    ]).sum()
    sac_flies = (events == 'sac_fly').sum()
    
    babip_numerator = hits - home_runs
    babip_denominator = at_bats - strikeouts - home_runs + sac_flies
    
    if babip_denominator > 0:
        metrics['babip'] = round(babip_numerator / babip_denominator, 3)
    else:
        metrics['babip'] = None
    
    # 計算 wOBA
    woba_weights = {
        'walk': 0.690,
        'hit_by_pitch': 0.722,
        'single': 0.883,
        'double': 1.244,
        'triple': 1.569,
        'home_run': 2.015
    }
    
    woba_numerator = 0
    for event, weight in woba_weights.items():
        count = (events == event).sum()
        woba_numerator += count * weight
    
    woba_denominator = at_bats + walks + (events == 'hit_by_pitch').sum() + sac_flies
    
    if woba_denominator > 0:
        metrics['woba'] = round(woba_numerator / woba_denominator, 3)
    else:
        metrics['woba'] = None
    
    return metrics


def _calculate_trend(early_val, mid_val, late_val) -> str:
    """計算趨勢方向"""
    if early_val is None or late_val is None:
        return "insufficient_data"
    
    diff = late_val - early_val
    if abs(diff) < 1:
        return "stable"
    elif diff > 0:
        return "increasing"
    else:
        return "decreasing"


def build_diagnosis_json(player_name: str, early_metrics: Dict, mid_metrics: Dict, late_metrics: Dict) -> Dict:
    """
    封裝三段資料為診斷 JSON
    """
    return {
        'player_name': player_name,
        'analysis_segments': {
            'early': early_metrics,
            'mid': mid_metrics,
            'late': late_metrics
        },
        'summary': {
            'total_games_analyzed': early_metrics['games'] + mid_metrics['games'] + late_metrics['games'],
            'launch_speed_trend': _calculate_trend(
                early_metrics.get('avg_launch_speed'),
                mid_metrics.get('avg_launch_speed'),
                late_metrics.get('avg_launch_speed')
            ),
            'hard_hit_trend': _calculate_trend(
                early_metrics.get('hard_hit_rate'),
                mid_metrics.get('hard_hit_rate'),
                late_metrics.get('hard_hit_rate')
            ),
            'k_rate_trend': _calculate_trend(
                early_metrics.get('k_rate'),
                mid_metrics.get('k_rate'),
                late_metrics.get('k_rate')
            )
        }
    }


def get_full_analysis(last_name: str, first_name: str, 
                      start_date: str = "2024-03-20", 
                      end_date: str = "2024-10-31") -> Dict:
    """
    完整分析流程：從球員姓名到診斷 JSON
    
    Args:
        last_name: 球員姓氏
        first_name: 球員名字
        start_date: 賽季開始日期
        end_date: 賽季結束日期
    
    Returns:
        Dict: 完整診斷 JSON
    """
    # Step 1: 獲取球員 ID
    player_id = get_player_id(last_name, first_name)
    player_name = f"{first_name} {last_name}"
    
    # Step 2: 抓取 Statcast 資料
    df = fetch_statcast_data(player_id, start_date, end_date)
    
    # Step 3: 切片
    early_df, mid_df, late_df = slice_by_game_index(df)
    
    # Step 4: 聚合指標
    early_metrics = aggregate_metrics(early_df, "Early (前10場)")
    mid_metrics = aggregate_metrics(mid_df, "Mid (季中10場)")
    late_metrics = aggregate_metrics(late_df, "Late (最後10場)")
    
    # Step 5: 封裝 JSON
    diagnosis = build_diagnosis_json(player_name, early_metrics, mid_metrics, late_metrics)
    
    return diagnosis
