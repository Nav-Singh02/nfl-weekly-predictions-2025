# src/data.py
from __future__ import annotations
import pandas as pd
import nfl_data_py as nfl

def load_schedule(seasons: list[int]) -> pd.DataFrame:
    """
    Return NFL schedules for the given seasons with consistent columns.
    Includes scores if games are already played.
    """
    sch = nfl.import_schedules(seasons)
    cols = ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score", "gameday"]
    sch = sch[cols].rename(columns={"gameday": "game_date"})
    return sch

def get_week_games(season: int, week: int) -> pd.DataFrame:
    """
    Return the schedule rows for one season/week (teams, scores if available).
    """
    sch = load_schedule([season])
    return sch[(sch["season"] == season) & (sch["week"] == week)].reset_index(drop=True)

# --- spread snapshot (pregame) ---
def get_week_spread_snapshot(season: int, week: int):
    """
    Returns a DataFrame with ['home_team','away_team','home_spread_snapshot'].
    Negative => home favored; Positive => home underdog.
    Rounded to 1 decimal.
    """
    sch = nfl.import_schedules([season])
    sch = sch[sch["week"] == week][["home_team", "away_team", "spread_line"]].copy()
    sch = sch.rename(columns={"spread_line": "home_spread_snapshot"})
    # Flip sign so home favorites are negative
    sch["home_spread_snapshot"] = (-pd.to_numeric(sch["home_spread_snapshot"], errors="coerce")).round(1)
    return sch


