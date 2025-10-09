# scripts/predict_week.py
import argparse, os, numpy as np, pandas as pd
import nfl_data_py as nfl         # past seasons (2015–season-1)
import nflreadpy as nfr           # current-season weekly player stats
from pathlib import Path

from src.data import get_week_games, load_schedule
from src.data import get_week_spread_snapshot
from src.features import (
    aggregate_weekly_to_team_stats,
    make_matchup_features,
    add_points_from_schedule,
    compute_team_rest_features,      # rest
    compute_team_travel_features,    # travel/tz/road
)
from src.models import fit_models, predict_proba
from src.ensemble import combine_probs
from src.elo import compute_elo_table, add_delta_elo

def _add_rest_deltas(feat_df: pd.DataFrame, sched_df: pd.DataFrame) -> pd.DataFrame:
    rest = compute_team_rest_features(sched_df)
    r_home = rest.rename(columns={
        "team":"home_team","rest_days":"home_rest_days",
        "short_week":"home_short_week","off_bye":"home_off_bye",
    })
    out = feat_df.merge(
        r_home[["season","week","home_team","home_rest_days","home_short_week","home_off_bye"]],
        on=["season","week","home_team"], how="left",
    )
    r_away = rest.rename(columns={
        "team":"away_team","rest_days":"away_rest_days",
        "short_week":"away_short_week","off_bye":"away_off_bye",
    })
    out = out.merge(
        r_away[["season","week","away_team","away_rest_days","away_short_week","away_off_bye"]],
        on=["season","week","away_team"], how="left",
    )
    out["delta_rest_days"]  = out["home_rest_days"].fillna(0)  - out["away_rest_days"].fillna(0)
    out["delta_short_week"] = out["home_short_week"].fillna(0) - out["away_short_week"].fillna(0)
    out["delta_off_bye"]    = out["home_off_bye"].fillna(0)    - out["away_off_bye"].fillna(0)
    return out

def _add_travel_deltas(feat_df: pd.DataFrame, sched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attaches travel/tz/road-streak and creates deltas:
      delta_tz_bucket, delta_b2b_road, delta_road_3in4
    """
    trav = compute_team_travel_features(sched_df)
    t_home = trav.rename(columns={
        "team":"home_team",
        "tz_bucket_vs_host":"home_tz_bucket",
        "b2b_road":"home_b2b_road",
        "road_3in4":"home_3in4_road",
    })
    out = feat_df.merge(
        t_home[["season","week","home_team","home_tz_bucket","home_b2b_road","home_3in4_road"]],
        on=["season","week","home_team"], how="left",
    )
    t_away = trav.rename(columns={
        "team":"away_team",
        "tz_bucket_vs_host":"away_tz_bucket",
        "b2b_road":"away_b2b_road",
        "road_3in4":"away_3in4_road",
    })
    out = out.merge(
        t_away[["season","week","away_team","away_tz_bucket","away_b2b_road","away_3in4_road"]],
        on=["season","week","away_team"], how="left",
    )
    out["delta_tz_bucket"]   = out["home_tz_bucket"].fillna(0)   - out["away_tz_bucket"].fillna(0)
    out["delta_b2b_road"]    = out["home_b2b_road"].fillna(0)    - out["away_b2b_road"].fillna(0)
    out["delta_road_3in4"]   = out["home_3in4_road"].fillna(0)   - out["away_3in4_road"].fillna(0)
    return out

def main(season: int, week: int, out_path: str):
    # === 1) TRAINING HISTORY: 2015..(season-1) ===
    training_years = list(range(2015, season))
    weekly_players = nfl.import_weekly_data(training_years)
    team_stats = aggregate_weekly_to_team_stats(weekly_players)

    # Schedules for training years (with scores & dates)
    sch = load_schedule(training_years)

    # Add actual points into team_stats before rolling features
    team_stats = add_points_from_schedule(team_stats, sch)

    # === Option A: expand FEATURE HISTORY with current-season weeks < target ===
    hist_team_stats = team_stats.copy()
    sch_for_elo = sch.copy()  # we'll extend this for Elo too

    if week > 1:
        try:
            cur_pl = nfr.load_player_stats([season])
            current_weekly = cur_pl.to_pandas()
            current_weekly = current_weekly[current_weekly["week"] < week]
            if not current_weekly.empty:
                team_stats_curr = aggregate_weekly_to_team_stats(current_weekly)

                sch_curr = load_schedule([season])
                sch_curr = sch_curr[sch_curr["week"] < week]
                team_stats_curr = add_points_from_schedule(team_stats_curr, sch_curr)

                hist_team_stats = pd.concat([hist_team_stats, team_stats_curr], ignore_index=True, sort=False)

                sch_curr_done = sch_curr.dropna(subset=["home_score","away_score"])
                if not sch_curr_done.empty:
                    sch_for_elo = pd.concat([sch_for_elo, sch_curr_done], ignore_index=True, sort=False)
        except Exception as e:
            print(f"Note: skipping current-season augmentation ({e}). Using past seasons only.")

    # === 2) TRAIN: base features + Elo + rest + travel ===
    train_feat = make_matchup_features(sch, team_stats, window=5).dropna()
    elo_table = compute_elo_table(sch_for_elo)
    train_feat = add_delta_elo(train_feat, elo_table)

    # NEW: add rest & travel deltas to training features
    train_feat = _add_rest_deltas(train_feat, sch)
    train_feat = _add_travel_deltas(train_feat, sch)

    # Build design matrix (all delta_* features)
    X_cols = [c for c in train_feat.columns if c.startswith("delta_")]
    X_train, y_train = train_feat[X_cols].fillna(0.0).values, train_feat["home_win"].values
    trio = fit_models(X_train, y_train)

    # === 3) PREDICT: build week features + Elo + rest + travel ===
    games = get_week_games(season, week)
    wk_feat = make_matchup_features(games, hist_team_stats, window=5)
    wk_feat = add_delta_elo(wk_feat, elo_table)

    sch_full = load_schedule([season])     # season schedule with dates
    wk_feat = _add_rest_deltas(wk_feat, sch_full)
    wk_feat = _add_travel_deltas(wk_feat, sch_full)

    # Predict
    X_wk = wk_feat[X_cols].fillna(0.0).values
    p_rf, p_xgb, p_lr = predict_proba(trio, X_wk)
    p = combine_probs(p_rf, p_xgb, p_lr)

    # === 4) Output CSV (UNCHANGED except spread snapshot) ===
    out = wk_feat[["season", "week", "home_team", "away_team"]].copy()
    out["Win Prob(Home)"] = p.round(2)
    out["Predicted Winner"] = np.where(
        p >= 0.5, out["home_team"].str.upper(), out["away_team"].str.upper()
    )

    # Merge spread snapshot and enforce column order
    snap = get_week_spread_snapshot(season, week)  # ['home_team','away_team','home_spread_snapshot']
    out = out.merge(snap, on=["home_team","away_team"], how="left")
    out = out[[
        "season", "week", "home_team", "away_team",
        "Win Prob(Home)", "home_spread_snapshot", "Predicted Winner"
    ]]
    out = out.sort_values(["Win Prob(Home)"], ascending=False).reset_index(drop=True)

    # === 5) Save CSV ===
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} predictions -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    out_path = args.out or f"predictions/{args.season}/week_{args.week}.csv"
    main(args.season, args.week, out_path)
