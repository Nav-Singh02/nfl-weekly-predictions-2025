# scripts/score_week.py
import argparse, pandas as pd, numpy as np
from pathlib import Path
from src.data import get_week_games

def score_week(season:int, week:int, preds_path:str, out_path:str):
    pred = pd.read_csv(preds_path)
    games = get_week_games(season, week)  # has home_score/away_score once final

    # Merge predictions with finals
    m = pred.merge(
        games[["home_team","away_team","home_score","away_score"]],
        on=["home_team","away_team"], how="left"
    ).dropna(subset=["home_score","away_score"]).copy()

    if m.empty:
        print("No completed games yet for this week.")
        return

    # Predicted home probability & actual outcome
    p = m["Win Prob(Home)"].astype(float)
    y = (m["home_score"] > m["away_score"]).astype(int)

    # Predicted pick: HOME vs AWAY
    pick_is_home = (m["Predicted Winner"] == m["home_team"].str.upper()).astype(int)

    # Metrics
    acc = float((pick_is_home == y).mean())
    brier = float(((p - y)**2).mean())

    # Per-game table
    out = m[["home_team","away_team","home_score","away_score","Win Prob(Home)","Predicted Winner"]].copy()
    out["Correct"] = np.where((pick_is_home == y), 1, 0)

    # Write per-game scored file
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # ---- Season metrics log (idempotent update) ----
    metrics_path = Path(f"predictions/{season}/metrics.csv")
    row = pd.DataFrame([{
        "season": season,
        "week": week,
        "completed_games": int(len(out)),
        "accuracy": acc,
        "brier": brier,
    }])

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        # drop existing row for this week/season if present, then append
        metrics = metrics[~((metrics["season"]==season) & (metrics["week"]==week))]
        metrics = pd.concat([metrics, row], ignore_index=True)
        metrics = metrics.sort_values(["season","week"]).reset_index(drop=True)
    else:
        metrics = row

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)

    print(f"Completed games: {len(out)}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Brier score: {brier:.3f}")
    print(f"Wrote details -> {out_path}")
    print(f"Updated metrics -> {metrics_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--preds", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    out_path = args.out or f"predictions/{args.season}/week_{args.week}_scored.csv"
    score_week(args.season, args.week, args.preds, out_path)
