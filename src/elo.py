# src/elo.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List

def compute_elo_table(
    schedules: pd.DataFrame,
    baseline: float = 1500.0,
    k: float = 20.0,
    hfa: float = 55.0,
    regress: float = 0.75,
) -> pd.DataFrame:
    """
    Build an Elo rating table per (season, week, team), including a 'week 0'
    row for each season after off-season regression. Uses only COMPLETED games.
    """
    cols_needed = {"season","week","home_team","away_team","home_score","away_score"}
    if not cols_needed.issubset(schedules.columns):
        missing = cols_needed - set(schedules.columns)
        raise ValueError(f"compute_elo_table: schedules missing columns: {missing}")

    sched = schedules.dropna(subset=["home_score","away_score"]).copy()
    seasons = sorted(sched["season"].unique().tolist())

    # carry ratings season-to-season with regression to the mean (baseline)
    prev_ratings: Dict[str, float] = {}
    rows: List[dict] = []

    for season in seasons:
        season_mask = sched["season"] == season
        ss = sched.loc[season_mask].copy()

        # team set for this season
        teams = sorted(set(ss["home_team"]).union(set(ss["away_team"])))

        # start-of-season regression
        ratings: Dict[str, float] = {}
        for t in teams:
            r_prev = prev_ratings.get(t, baseline)
            ratings[t] = baseline + (r_prev - baseline) * regress

        # record week 0 rating for every team
        for t in teams:
            rows.append({"season": season, "week": 0, "team": t, "elo": ratings[t]})

        # iterate week-by-week
        for wk in sorted(ss["week"].unique().tolist()):
            wk_games = ss.loc[ss["week"] == wk]

            # process games in this week
            for _, g in wk_games.iterrows():
                ht, at = g["home_team"], g["away_team"]
                hr, ar = ratings.get(ht, baseline), ratings.get(at, baseline)

                # expected home win prob using Elo diff + home-field advantage
                expected_home = 1.0 / (1.0 + 10.0 ** (-( (hr + hfa - ar) / 400.0 )))
                # outcome
                if g["home_score"] > g["away_score"]:
                    outcome = 1.0
                elif g["home_score"] < g["away_score"]:
                    outcome = 0.0
                else:
                    outcome = 0.5

                delta = k * (outcome - expected_home)
                ratings[ht] = hr + delta
                ratings[at] = ar - delta

            # after the week, record ratings for all teams (byes carry through)
            for t in teams:
                rows.append({"season": season, "week": wk, "team": t, "elo": ratings[t]})

        prev_ratings = ratings  # carry forward

    elo = pd.DataFrame(rows).sort_values(["season","week","team"]).reset_index(drop=True)
    return elo

def add_delta_elo(games: pd.DataFrame, elo_table: pd.DataFrame, baseline: float = 1500.0) -> pd.DataFrame:
    """
    For each game row (season, week, home_team, away_team), attach
    delta_elo = Elo(home, prev_week) - Elo(away, prev_week).
    Relies on week 0 rows existing in elo_table.
    """
    out = games.copy()
    out["prev_week"] = out["week"] - 1

    home = elo_table.rename(columns={"team":"home_team", "elo":"home_elo", "week":"prev_week"})
    away = elo_table.rename(columns={"team":"away_team", "elo":"away_elo", "week":"prev_week"})

    out = out.merge(home[["season","prev_week","home_team","home_elo"]],
                    on=["season","prev_week","home_team"], how="left")
    out = out.merge(away[["season","prev_week","away_team","away_elo"]],
                    on=["season","prev_week","away_team"], how="left")

    out["home_elo"] = out["home_elo"].fillna(baseline)
    out["away_elo"] = out["away_elo"].fillna(baseline)
    out["delta_elo"] = out["home_elo"] - out["away_elo"]

    # clean helper
    return out.drop(columns=["prev_week"], errors="ignore")
