# src/features.py
from __future__ import annotations
import pandas as pd
import numpy as np

# Rolling per-game stats computed from PRIOR games only (shifted by 1 to avoid leakage)
# Includes: turnover_diff and points
NUMERIC_COLS = [
    "pass_yds",
    "rush_yds",
    "pass_td",
    "rush_td",
    "ints",
    "fumbles_lost",
    "turnover_diff",
    "points",
]

def _team_roll_means(team_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """For a single team-season, compute rolling per-game means over the last `window` games,
    shifted by 1 so the current game never sees its own stats."""
    team_df = team_df.sort_values(["season", "week"]).copy()
    for col in NUMERIC_COLS:
        team_df[f"{col}_pg"] = (
            team_df[col]
            .rolling(window, min_periods=1)
            .mean()
            .shift(1)
        )
    return team_df

def _first_existing(df: pd.DataFrame, candidates: list[str], fill_value=0) -> pd.Series:
    """Return the first existing column as a Series; otherwise a fill_value Series."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(fill_value, index=df.index, dtype="float64")

def aggregate_weekly_to_team_stats(weekly: pd.DataFrame) -> pd.DataFrame:
    """Convert player-level weekly rows into team-level per-game totals.
       Accepts schemas from both nfl_data_py and nflreadpy."""
    df = weekly.copy()

    # --- schema-flexible team/opponent detection ---
    if "recent_team" in df.columns:
        team_col = "recent_team"
    elif "team" in df.columns:
        team_col = "team"
    elif "posteam" in df.columns:
        team_col = "posteam"
    else:
        raise ValueError("Cannot find team column (expected 'recent_team', 'team', or 'posteam').")

    for cand in ["opponent_team", "opponent", "defteam", "def_team"]:
        if cand in df.columns:
            opp_col = cand
            break
    else:
        raise ValueError("Cannot find opponent column (expected one of: 'opponent_team','opponent','defteam','def_team').")

    df["team"] = df[team_col]
    df["opponent"] = df[opp_col]

    # Build a single fumbles_lost from available components (missing cols default to 0)
    for col in ["rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"]:
        if col not in df.columns:
            df[col] = 0

    df["fumbles_lost"] = (
        df["rushing_fumbles_lost"].fillna(0)
        + df["receiving_fumbles_lost"].fillna(0)
        + df["sack_fumbles_lost"].fillna(0)
    )

    # --- normalize core numeric columns across possible schemas ---
    df["_pass_yds"] = _first_existing(df, ["passing_yards", "pass_yards"])
    df["_rush_yds"] = _first_existing(df, ["rushing_yards", "rush_yards"])
    df["_pass_td"]  = _first_existing(df, ["passing_tds", "pass_touchdowns"])
    df["_rush_td"]  = _first_existing(df, ["rushing_tds", "rush_touchdowns"])
    df["_ints"]     = _first_existing(df, ["interceptions", "passing_interceptions"])

    # Ensure numeric
    for c in ["_pass_yds","_rush_yds","_pass_td","_rush_td","_ints","fumbles_lost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Aggregate player rows to team-per-game totals
    agg = df.groupby(["season", "week", "team", "opponent"], as_index=False).agg(
        pass_yds=("_pass_yds", "sum"),
        rush_yds=("_rush_yds", "sum"),
        pass_td =("_pass_td",  "sum"),
        rush_td =("_rush_td",  "sum"),
        ints    =("_ints",     "sum"),
        fumbles_lost=("fumbles_lost", "sum"),
    )

    # turnover differential (higher is better for the offense)
    agg["turnover_diff"] = -(agg["ints"] + agg["fumbles_lost"])

    # placeholder for points (merged later)
    if "points" not in agg.columns:
        agg["points"] = 0.0

    return agg

def add_points_from_schedule(team_stats: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Merge team points from schedules into team_stats by (season, week, team, opponent)."""
    s = schedules[["season", "week", "home_team", "away_team", "home_score", "away_score"]].copy()
    home = s.rename(columns={"home_team": "team", "away_team": "opponent", "home_score": "points"})[
        ["season", "week", "team", "opponent", "points"]
    ]
    away = s.rename(columns={"away_team": "team", "home_team": "opponent", "away_score": "points"})[
        ["season", "week", "team", "opponent", "points"]
    ]
    pts = pd.concat([home, away], ignore_index=True)
    out = team_stats.drop(columns=["points"], errors="ignore").merge(
        pts, on=["season", "week", "team", "opponent"], how="left"
    )
    out["points"] = out["points"].fillna(0.0)
    return out

# --- REST FEATURES (per team per game) ---
def compute_team_rest_features(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: schedule_df with at least ['season','week','game_date','home_team','away_team'].
    Output: DataFrame with per (season, week, team) columns:
        ['season','week','team','rest_days','short_week','off_bye']
    Rules:
      - rest_days = days since team's previous game date (NaN if none)
      - short_week = 1 if rest_days <= 6 else 0 (0 if NaN)
      - off_bye = 1 if team had no game in prior week (or rest_days >= 13); 0 if played last week
    """
    sch = schedule_df.copy()
    sch = sch.sort_values(['season', 'week']).reset_index(drop=True)

    # Long form: one row per team per game
    home = sch[['season','week','game_date','home_team']].rename(columns={'home_team':'team'})
    away = sch[['season','week','game_date','away_team']].rename(columns={'away_team':'team'})
    tg = pd.concat([home, away], ignore_index=True).sort_values(['team','season','week']).reset_index(drop=True)

    # Previous game info per team
    tg['prev_game_date'] = tg.groupby('team')['game_date'].shift(1)
    tg['prev_week'] = tg.groupby('team')['week'].shift(1)
    tg['rest_days'] = (pd.to_datetime(tg['game_date']) - pd.to_datetime(tg['prev_game_date'])).dt.days

    # Flags
    tg['short_week'] = (tg['rest_days'] <= 6).astype('Int64').fillna(0).astype(int)
    tg['played_last_week'] = (tg['prev_week'] == (tg['week'] - 1)).fillna(False)
    tg['off_bye'] = ((~tg['played_last_week']) | (tg['rest_days'] >= 13)).astype(int)

    return tg[['season','week','team','rest_days','short_week','off_bye']].copy()

# --- TRAVEL/TZ/ROAD STREAK FEATURES ---
TEAM_TZ = {
    # Eastern
    "BUF":"ET","MIA":"ET","NE":"ET","NYJ":"ET","BAL":"ET","CIN":"ET","CLE":"ET","PIT":"ET",
    "IND":"ET","JAX":"ET","HOU":"CT","TEN":"CT",
    "DAL":"CT","NYG":"ET","PHI":"ET","WAS":"ET","DET":"ET",
    "ATL":"ET","CAR":"ET","TB":"ET",
    # Central
    "CHI":"CT","GB":"CT","MIN":"CT","NO":"CT","KC":"CT",
    # Mountain / Pacific
    "DEN":"MT","ARI":"MT","SEA":"PT","SF":"PT","LA":"PT","LAC":"PT","LV":"PT"
}
TZ_OFFSET = {"ET": 0, "CT": -1, "MT": -2, "PT": -3}

def _team_tz(team: str) -> int:
    tz = TEAM_TZ.get(team)
    return TZ_OFFSET.get(tz, 0)

def compute_team_travel_features(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per (season, week, team) travel features:
      ['season','week','team','tz_bucket_vs_host','b2b_road','road_3in4']
    tz_bucket_vs_host: 0 if same tz as host team's base tz, 1 if 1-hr diff, 2 if >=2 hrs.
    b2b_road: 1 if team was away last week and is away this week.
    road_3in4: 1 if team plays 3 road games in its last 4 (including this one).
    """
    sch = schedule_df.copy().sort_values(["season","week"]).reset_index(drop=True)

    # Long form: one row per team per game with location
    home = sch[["season","week","game_date","home_team","away_team"]].copy()
    home = home.rename(columns={"home_team":"team"})
    home["is_away"] = 0
    home["host_team"] = home["team"]

    away = sch[["season","week","game_date","home_team","away_team"]].copy()
    away = away.rename(columns={"away_team":"team"})
    away["is_away"] = 1
    away["host_team"] = away["home_team"]

    tg = pd.concat([home, away], ignore_index=True)
    tg = tg.sort_values(["team","season","week"]).reset_index(drop=True)

    # Time-zone bucket relative to host (uses static team base tz as proxy for stadium tz)
    tg["tz_diff_hours"] = (tg["host_team"].map(_team_tz) - tg["team"].map(_team_tz)).abs()
    tg["tz_bucket_vs_host"] = tg["tz_diff_hours"].clip(0, 2)  # 0,1,2 only

    # Rolling road counts for streaks
    tg["prev_is_away"] = tg.groupby("team")["is_away"].shift(1).fillna(0).astype(int)
    tg["road_in_last4"] = (
        tg.groupby("team")["is_away"]
          .rolling(window=4, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
          .astype(int)
    )
    # Include this week's away in the rolling window above; flag for 3 in last 4 (including this one)
    tg["road_3in4"] = (tg["road_in_last4"] >= 3).astype(int)

    # Back-to-back road: today away AND last week away
    tg["b2b_road"] = ((tg["is_away"] == 1) & (tg["prev_is_away"] == 1)).astype(int)

    return tg[["season","week","team","tz_bucket_vs_host","b2b_road","road_3in4"]].copy()

def make_matchup_features(games: pd.DataFrame, team_stats: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Build a matchup row per game with home vs away rolling means and deltas, with prev-week alignment and week-1 fallback."""
    # 1) Rolling means per (team, season)
    rolled = (
        team_stats
        .groupby(["team", "season"], group_keys=False)
        .apply(lambda df: _team_roll_means(df, window))
        .reset_index(drop=True)
    )

    pg_cols = [f"{c}_pg" for c in NUMERIC_COLS]

    # 2) Lookups:
    #    - exact: (team, season, week) rolling rows
    #    - latest_non_na: last row per team that has any non-null rolling values
    exact = rolled[["team", "season", "week"] + pg_cols].copy()

    valid = rolled.dropna(subset=pg_cols, how="all")
    latest = (
        valid.sort_values(["team", "season", "week"])
             .groupby("team", as_index=False)
             .tail(1)[["team"] + pg_cols]
    )

    # 3) Align games to PREVIOUS week for rolling stats
    g = games.copy()
    g["prev_week"] = g["week"] - 1

    home_exact = exact.rename(columns={c: f"home_{c}" for c in exact.columns})
    away_exact = exact.rename(columns={c: f"away_{c}" for c in exact.columns})

    feat = (
        g
        .merge(
            home_exact,
            left_on=["home_team", "season", "prev_week"],
            right_on=["home_team", "home_season", "home_week"],
            how="left",
        )
        .merge(
            away_exact,
            left_on=["away_team", "season", "prev_week"],
            right_on=["away_team", "away_season", "away_week"],
            how="left",
        )
    )

    # 4) Fallback: fill missing with latest NON-NA rolling per team
    home_latest = latest.rename(columns={c: f"home_{c}" for c in latest.columns})
    away_latest = latest.rename(columns={c: f"away_{c}" for c in latest.columns})

    if feat["home_pass_yds_pg"].isna().any():
        feat = feat.merge(home_latest, on="home_team", how="left", suffixes=("", "_fallback"))
        for base in pg_cols:
            feat[f"home_{base}"] = feat[f"home_{base}"].fillna(feat[f"home_{base}_fallback"])
        feat = feat.drop(columns=[c for c in feat.columns if c.endswith("_fallback")], errors="ignore")

    if feat["away_pass_yds_pg"].isna().any():
        feat = feat.merge(away_latest, on="away_team", how="left", suffixes=("", "_fallback"))
        for base in pg_cols:
            feat[f"away_{base}"] = feat[f"away_{base}"].fillna(feat[f"away_{base}_fallback"])
        feat = feat.drop(columns=[c for c in feat.columns if c.endswith("_fallback")], errors="ignore")

    # 5) Differential features: home_pg - away_pg
    for base in pg_cols:
        feat[f"delta_{base}"] = feat[f"home_{base}"] - feat[f"away_{base}"]

    # 6) Optional label
    if {"home_score", "away_score"}.issubset(games.columns):
        feat["home_win"] = (feat["home_score"] > feat["away_score"]).astype(int)

    keep_meta = ["game_id", "season", "week", "home_team", "away_team"]
    keep_model = [c for c in feat.columns if c.startswith("delta_")]
    keep_target = ["home_win"] if "home_win" in feat.columns else []
    out = feat[keep_meta + keep_model + keep_target].copy()
    return out
