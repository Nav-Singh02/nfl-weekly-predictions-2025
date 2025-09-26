# scripts/predict_week.py
import argparse, os, numpy as np, pandas as pd
import nfl_data_py as nfl         # keep for past seasons (2015–2024)
import nflreadpy as nfr           # NEW: use for current-season weekly data
from pathlib import Path

from src.data import get_week_games, load_schedule
from src.features import (
    aggregate_weekly_to_team_stats,
    make_matchup_features,
    add_points_from_schedule,
)
from src.models import fit_models, predict_proba
from src.ensemble import combine_probs

def main(season: int, week: int, out_path: str):
    # === 1) TRAINING HISTORY: 2015..(season-1) via nfl_data_py ===
    training_years = list(range(2015, season))
    weekly_players = nfl.import_weekly_data(training_years)
    team_stats = aggregate_weekly_to_team_stats(weekly_players)

    # Labeled schedules for training years (also via nfl_data_py)
    sch = load_schedule(training_years)

    # Add actual points into team_stats before rolling features
    team_stats = add_points_from_schedule(team_stats, sch)

    # === Option A: augment FEATURE HISTORY with current-season weeks < target (via nflreadpy) ===
    hist_team_stats = team_stats.copy()
    if week > 1:
        try:
            # nflreadpy returns a Polars DataFrame; convert to pandas
            cur_pl = nfr.load_player_stats([season])   # week-level player stats
            current_weekly = cur_pl.to_pandas()
            current_weekly = current_weekly[current_weekly["week"] < week]

            if not current_weekly.empty:
                team_stats_curr = aggregate_weekly_to_team_stats(current_weekly)

                # Use our existing schedules helper for points merge
                sch_curr = load_schedule([season])
                sch_curr = sch_curr[sch_curr["week"] < week]
                team_stats_curr = add_points_from_schedule(team_stats_curr, sch_curr)

                hist_team_stats = pd.concat(
                    [hist_team_stats, team_stats_curr], ignore_index=True, sort=False
                )
        except Exception as e:
            print(f"Note: skipping current-season augmentation ({e}). Using past seasons only.")

    # === 2) Train ensemble on past seasons only ===
    train_feat = make_matchup_features(sch, team_stats, window=5).dropna()
    X_cols = [c for c in train_feat.columns if c.startswith("delta_")]
    X_train, y_train = train_feat[X_cols].values, train_feat["home_win"].values
    trio = fit_models(X_train, y_train)

    # === 3) Predict target week using augmented feature base ===
    games = get_week_games(season, week)
    wk_feat = make_matchup_features(games, hist_team_stats, window=5)
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
