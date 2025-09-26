# NFL_Game_Prediction_Models
🏈 NFL Predictions 2025 — Machine Learning Model

Welcome to the NFL Game Prediction Models repository! This project uses machine learning, NFL data APIs, and historical game results to predict NFL game outcomes for the 2025 season and beyond.

## 🚀 Project Overview
This repository contains an **ensemble** machine learning system that predicts NFL game results based on team performance metrics, historical data, and lightweight statistical modeling. The model leverages:

- **nfl_data_py** for historical schedules/scores (2015–2024)
- **nflreadpy** for current-season weekly player stats (converted to team-game totals)
- **Advanced feature engineering** (rolling deltas, turnover differential, points, and **Elo**)
- **Ensemble learning** combining Random Forest, XGBoost, and Logistic Regression
- **Isotonic calibration** so probabilities are well-calibrated (not overconfident)
- Clean, reproducible code in `src/` + `scripts/` (no betting lines)

## 📊 Data Sources
- **nfl_data_py**: NFL schedules and historical results (training set: 2015–2024)
- **nflreadpy**: Current-season weekly player stats, aggregated to team-level per game
- **Engineered metrics**: rolling per-game stats, turnover differential, points from schedules, and **Elo** with offseason regression + home-field advantage

## 🏈 How It Works
1. **Data Collection**: Pull historical schedules/results (2015–2024) with `nfl_data_py`. For the current season, load weekly player stats with `nflreadpy` and aggregate to team-game totals for weeks **before** the target week.
2. **Preprocessing & Feature Engineering**: Build leakage-safe rolling means (5-game window, shifted by 1) for passing/rushing yards & TDs, interceptions, fumbles lost; compute turnover differential and merge points from schedules; compute per-team **Elo** and attach `delta_elo` (home − away).
3. **Model Training**: Train **Random Forest**, **XGBoost**, and **Logistic Regression** on historical games; calibrate each model’s probabilities using **isotonic** (cv=5).
4. **Prediction**: For a given week, generate a CSV with:
   - `season, week, home_team, away_team, Win Prob(Home), Predicted Winner`
5. **Evaluation**: After games finish, score the week using **Accuracy** and **Brier score**; append metrics to `predictions/2025/metrics.csv`.

## Dependencies
- nfl_data_py  
- nflreadpy  
- xgboost  
- scikit-learn  
- pandas, numpy  
- pyarrow, polars, tqdm  
- (optional) matplotlib

## 📈 Model Performance
- **Accuracy**: share of correct picks each week  
- **Brier score**: mean squared error of probabilities (0 = perfect; coin-flip ≈ 0.25)  
- Example (from this repo): **Week 1** — Accuracy **62.5%**, Brier **0.233**  
- All weekly metrics are logged to `predictions/2025/metrics.csv` and updated as weeks are scored

## 🔑 Key Features Used
- **Rolling team form** (5-game window, leakage-safe via `.shift(1)`)
  - Passing yards & TDs per game  
  - Rushing yards & TDs per game  
  - Interceptions thrown per game  
  - Fumbles lost per game
- **Turnover differential** (−INT − fumbles lost; offense-centric)
- **Points scored** merged from official schedules (actual results)
- **Elo team strength** with offseason regression & home-field advantage  
  - Model feature: **`delta_elo`** = home Elo − away Elo (aligned to the **previous** week)
- **Modeling**
  - Ensemble: **Random Forest + XGBoost + Logistic Regression**  
  - **Isotonic calibration** (cv=5) for well-calibrated win probabilities
- **No betting lines**; current-season context uses only weeks **before** the target week

## 📌 Future Improvements
- Incorporate **injury reports/inactives** and QB availability
- Add **opponent defensive splits** and QB-centric efficiency metrics
- Model **rest/travel** (short weeks, time zones) and **weather** for outdoor games
- Include **special teams** and **penalties** features
- Hyperparameter tuning & per-season reweighting; per-week **isotonic refits**
- Optional benchmarking vs. **betting lines** (kept out in this version)
- **Automation**: GitHub Actions to auto-predict Tuesdays and score Mondays
- Simple dashboard/notebook to visualize weekly accuracy and calibration

## 📜 License
Licensed under the **MIT License**.  
This project **extends ideas from** Sujar Henry’s public work on weekly NFL predictions; all code in `src/` and `scripts/` is my own implementation organized for a reproducible pipeline.
