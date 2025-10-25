from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import os

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

def predict_sleep_duration(model, input_data):
    return model.predict(input_data)

def recommend_bedtime(wake_time, predicted_duration, buffer_minutes = 15):
    return None