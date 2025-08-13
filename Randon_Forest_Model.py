# Load the car data and prepare for enhanced dashboard with car listing and prediction
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import json

# Load the data
df = pd.read_csv('car data.csv')

# Prepare data for machine learning model
df_ml = df.copy()

# Create label encoders for categorical variables
le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_transmission = LabelEncoder()

df_ml['Fuel_Type_encoded'] = le_fuel.fit_transform(df_ml['Fuel_Type'])
df_ml['Selling_type_encoded'] = le_seller.fit_transform(df_ml['Selling_type'])
df_ml['Transmission_encoded'] = le_transmission.fit_transform(df_ml['Transmission'])

# Prepare features for ML model
X = df_ml[['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type_encoded', 'Selling_type_encoded', 'Transmission_encoded', 'Owner']]
y = df_ml['Selling_Price']

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importance for display
feature_importance = dict(zip(['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner'],
                             model.feature_importances_))

print("Machine Learning Model Trained!")
print("Model Score:", round(model.score(X, y), 3))
print("Feature Importance:")
for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {round(importance, 3)}")

# Prepare car data for JavaScript (convert to JSON-friendly format)
cars_data = []
for idx, row in df.iterrows():
    cars_data.append({
        'name': row['Car_Name'],
        'year': int(row['Year']),
        'selling_price': float(row['Selling_Price']),
        'present_price': float(row['Present_Price']),
        'driven_kms': int(row['Driven_kms']),
        'fuel_type': row['Fuel_Type'],
        'selling_type': row['Selling_type'],
        'transmission': row['Transmission'],
        'owner': int(row['Owner'])
    })

print(f"\
Prepared {len(cars_data)} cars for the dashboard")
print("Sample car data:", cars_data[0])