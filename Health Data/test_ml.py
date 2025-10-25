import numpy as np
import pandas as pd
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Load data
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('HealthData.csv')
print(df.head())
print(df.info())

#Convert time strings to float hours
def convert_time(duration):
    if pd.isna(duration):
        return np.nan
    
    split = duration.strip().replace('h', ' ').replace('min', '').split()
    hours = int(split[0])
    if len(split) > 1:
        minutes = int(split[1])

    else:
        minutes = 0

    return hours + minutes / 60.0

#Process data to float hours
df['Sleep Duration'] = df['Avg Duration'].apply(convert_time)
df['Sleep Need'] = df['Avg Sleep Need'].apply(convert_time)

#Create lag features for time series modeling
df['Duration_lag1'] = df['Sleep Duration'].shift(1)
df['Duration_lag2'] = df['Sleep Duration'].shift(2)
df['Score_lag1'] = df['Avg Score'].shift(1)
df['Stress_lag1'] = df['Stress'].shift(1)
df['Duration_roll_avg'] = df['Sleep Duration'].shift(1).rolling(window=3, min_periods=1).mean()
df['Stress_rolling_avg'] = df['Stress'].shift(1).rolling(window=3, min_periods=1).mean() 

features = ['Avg Score', 'Stress', 'Avg Resting Heart Rate', 'Avg High Heart Rate', 'Steps', 'Intensity Minutes', 'Sleep Need',
            'Duration_lag1', 'Duration_lag2', 'Score_lag1', 'Stress_lag1',
            'Duration_roll_avg', 'Stress_rolling_avg'
           ]

data = df[features + ['Sleep Duration']].dropna()
x = data[features]
y = data['Sleep Duration']

models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5, min_samples_split=5),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
    'Ridge Regression': Ridge(alpha=1.0)
}
# Test moodels with TimeSeriesSplit cross-validation
result = []
important = {}

for name, model in models.items():
    scores = []
    maes = []
    impford = []

    for fold, (train_idx, test_idx) in enumerate(TimeSeriesSplit(n_splits=5).split(x), 1):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(x_train, y_train)
        predict = model.predict(x_test)
        r2 = r2_score(y_test, predict)
        mae = mean_absolute_error(y_test, predict)
        scores.append(r2)
        maes.append(mae)

        if hasattr(model, 'feature_importances_'):
            impford.append(model.feature_importances_)

    avg_r2 = np.mean(scores)
    avg_mae = np.mean(maes)
    result.append({'Model': name, 'R²': avg_r2, 'Error_min': avg_mae*60})
    if impford:
        important[name] = np.mean(impford, axis=0)

    elif hasattr(model, 'coef_'):
        important[name] = np.abs(model.coef_)

results_df = pd.DataFrame(result).sort_values('R²', ascending=False)
print("\n" + results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"Best Model: {best['Model']}")
print(f"Prediction Accuracy: R²={best['R²']:.3f}")
print(f"Average Error: ±{best['Error_min']:.0f} minutes")
