# scripts/predict_week.py
import argparse, os, numpy as np, pandas as pd
import nfl_data_py as nfl         # past seasons (2015–season-1)
import nflreadpy as nfr           # current-season weekly player stats
from pathlib import Path

from src.data import get_week_games, load_schedule
from src.features import (
    aggregate_weekly_to_team_stats,
    make_matchup_features,
    add_points_from_schedule,
)
from src.models import fit_models, predict_proba
from src.ensemble import combine_probs
from src.elo import compute_elo_table, add_delta_elo   # <-- NEW

def main(season: int, week: int, out_path: str):
    # === 1) TRAINING HISTORY: 2015..(season-1) ===
    training_years = list(range(2015, season))
    weekly_players = nfl.import_weekly_data(training_years)
    team_stats = aggregate_weekly_to_team_stats(weekly_players)

    # Schedules for training years (with scores)
    sch = load_schedule(training_years)

    # Add actual points into team_stats before rolling features
    team_stats = add_points_from_schedule(team_stats, sch)

    # === Option A: expand FEATURE HISTORY with current-season weeks < target ===
    hist_team_stats = team_stats.copy()
    sch_for_elo = sch.copy()  # we'll extend this for Elo too

    if week > 1:
        try:
            # Current season player-week rows up to week-1
            cur_pl = nfr.load_player_stats([season])
            current_weekly = cur_pl.to_pandas()
            current_weekly = current_weekly[current_weekly["week"] < week]
            if not current_weekly.empty:
                team_stats_curr = aggregate_weekly_to_team_stats(current_weekly)

                # Current season schedule up to week-1 (for points & Elo)
                sch_curr = load_schedule([season])
                sch_curr = sch_curr[sch_curr["week"] < week]
                team_stats_curr = add_points_from_schedule(team_stats_curr, sch_curr)

                hist_team_stats = pd.concat([hist_team_stats, team_stats_curr], ignore_index=True, sort=False)

                # extend Elo history with completed current-season games (< week)
                sch_curr_done = sch_curr.dropna(subset=["home_score","away_score"])
                if not sch_curr_done.empty:
                    sch_for_elo = pd.concat([sch_for_elo, sch_curr_done], ignore_index=True, sort=False)
        except Exception as e:
            print(f"Note: skipping current-season augmentation ({e}). Using past seasons only.")

    # === 2) TRAIN: build features on past seasons only, then attach Elo delta ===
    train_feat = make_matchup_features(sch, team_stats, window=5).dropna()
    # Compute Elo on schedules with real results (past seasons + completed current-season weeks)
    elo_table = compute_elo_table(sch_for_elo)
    train_feat = add_delta_elo(train_feat, elo_table)   # adds 'delta_elo'

    X_cols = [c for c in train_feat.columns if c.startswith("delta_")]
    X_train, y_train = train_feat[X_cols].values, train_feat["home_win"].values
    trio = fit_models(X_train, y_train)

    # === 3) PREDICT: build week features from expanded history, then attach Elo delta ===
    games = get_week_games(season, week)
    wk_feat = make_matchup_features(games, hist_team_stats, window=5)
    wk_feat = add_delta_elo(wk_feat, elo_table)         # adds 'delta_elo' to target week

    X_wk = wk_feat[X_cols].fillna(0.0).values
    p_rf, p_xgb, p_lr = predict_proba(trio, X_wk)
    p = combine_probs(p_rf, p_xgb, p_lr)

    # === 4) Clean output ===
    out = wk_feat[["season", "week", "home_team", "away_team"]].copy()
    out["Win Prob(Home)"] = p.round(2)
    out["Predicted Winner"] = np.where(
        p >= 0.5, out["home_team"].str.upper(), out["away_team"].str.upper()
    )
    out = out[["season", "week", "home_team", "away_team", "Win Prob(Home)", "Predicted Winner"]]
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
