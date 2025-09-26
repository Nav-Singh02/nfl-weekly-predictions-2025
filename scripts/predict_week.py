# scripts/predict_week.py
import argparse, os, numpy as np, nfl_data_py as nfl
from pathlib import Path
from src.data import get_week_games, load_schedule
from src.features import aggregate_weekly_to_team_stats, make_matchup_features
from src.models import fit_models, predict_proba
from src.ensemble import combine_probs

def main(season: int, week: int, out_path: str):
    # 1) History to train on: 2015 up to the season *before* the target season
    training_years = list(range(2015, season))
    weekly_players = nfl.import_weekly_data(training_years)
    team_stats = aggregate_weekly_to_team_stats(weekly_players)

    # Labeled training rows from schedules (same training_years)
    sch = load_schedule(training_years)
    train_feat = make_matchup_features(sch, team_stats, window=5).dropna()
    X_cols = [c for c in train_feat.columns if c.startswith("delta_")]
    X_train, y_train = train_feat[X_cols].values, train_feat["home_win"].values

    # 2) Fit ensemble (RF + XGB + LR)
    trio = fit_models(X_train, y_train)

    # 3) Predict target week
    games = get_week_games(season, week)
    wk_feat = make_matchup_features(games, team_stats, window=5)
    X_wk = wk_feat[X_cols].fillna(0.0).values
    p_rf, p_xgb, p_lr = predict_proba(trio, X_wk)
    p = combine_probs(p_rf, p_xgb, p_lr)

    # 4) Clean, human-readable output
    out = wk_feat[["season", "week", "home_team", "away_team"]].copy()
    out["Win Prob(Home)"] = p.round(2)
    out["Predicted Winner"] = np.where(
        p >= 0.5,
        out["home_team"].str.upper(),
        out["away_team"].str.upper()
    )
    out = out[["season", "week", "home_team", "away_team", "Win Prob(Home)", "Predicted Winner"]]
    # Optional: keep most confident picks at the top
    out = out.sort_values(["Win Prob(Home)"], ascending=False).reset_index(drop=True)

    # 5) Save CSV
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} predictions -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    # default: predictions/<season>/week_<week>.csv
    out_path = args.out or f"predictions/{args.season}/week_{args.week}.csv"
    main(args.season, args.week, out_path)
