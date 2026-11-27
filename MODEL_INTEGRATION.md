# CatBoost model integration

## Files changed
- `src/feature_engineering.py`: mirrors the feature generation in the team's supplied CatBoost training script.
- `src/predict.py`: loads `models/catboost_model_2w.joblib` / `4w.joblib`, builds the latest ward row and returns the dashboard response.
- `src/risk_engine.py`: converts a numerical forecast into a transparent ward-relative percentile band.
- `main.py`: adds authenticated, CSRF-protected `POST /api/predict`.
- `static/js/user/user.js`: calls the API and renders the response.
- `templates/user/user_dashboard.html`: exposes the CSRF token and clarifies historical-data status.

## Required model files
The repository ZIP does not contain trained binary model files. Copy the three trained artifacts from your working project into this project's `models/` folder. The dashboard needs the 2-week and 4-week files:

```text
models/catboost_model_2w.joblib
models/catboost_model_3w.joblib
models/catboost_model_4w.joblib
```

## Run locally
1. Ensure `.env` exists and has a valid `SESSION_SECRET` (at least 32 characters) and database settings.
2. Install dependencies: `pip install -r requirements.txt`.
3. Start from the repository root: `uvicorn main:app --reload`.
4. Sign in as a hospital user and open `/user_dashboard`.
5. Select a city, ward and 2 or 4 weeks, then generate the forecast.

## Important data/model caveats
- The processed dataset in the supplied repository ends on **2025-12-29**. The endpoint forecasts from that last historical observation; it does not fetch live weather or live dengue reports. The returned date is the target date relative to the final observation, not today's date.
- The risk categories use each ward's historical case-count percentiles (50th, 75th and 90th). These are prototype-relative labels, not medically validated alert thresholds. Calibrate them with public-health authorities before operational use.
- Driver text describes the latest observed rainfall, temperature, humidity and cases relative to historical values. It is not a SHAP explanation of model contribution.
- The training script's feature schema must remain unchanged. If features are edited, retrain and regenerate the saved model bundles.
- The `3w` model is retained but the supplied dashboard offers only 2-week and 4-week selection.

## Smoke test
After starting the app and signing in, use the dashboard. The prediction endpoint requires a valid authenticated hospital-user session and `X-CSRF-Token` from the page's meta tag. A direct unauthenticated API call is expected to fail.
