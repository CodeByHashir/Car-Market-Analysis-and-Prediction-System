# Create the main data processing script
data_processing_script = '''"""
Car Market Analytics - Data Processing Script
============================================

This script handles data loading, cleaning, preprocessing, and model training
for the Car Market Analytics Dashboard project.

Author: Car Analytics Team
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class CarDataProcessor:
    """
    Main class for processing car market data and training prediction models.
    """

    def __init__(self, data_file='car data.csv'):
        """
        Initialize the data processor.

        Args:
            data_file (str): Path to the car data CSV file
        """
        self.data_file = data_file
        self.df = None
        self.model = None
        self.label_encoders = {}
        self.feature_columns = ['Year', 'Present_Price', 'Driven_kms', 
                               'Fuel_Type_encoded', 'Selling_type_encoded', 
                               'Transmission_encoded', 'Owner']

    def load_data(self):
        """Load and display basic information about the dataset."""
        try:
            self.df = pd.read_csv(self.data_file, encoding='ascii')
            print(f"Data loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def explore_data(self):
        """Perform exploratory data analysis."""
        if self.df is None:
            print("Please load data first!")
            return

        print("\\
=== DATASET OVERVIEW ===")
        print(self.df.head())

        print("\\
=== DATA TYPES ===")
        print(self.df.dtypes)

        print("\\
=== MISSING VALUES ===")
        print(self.df.isnull().sum())

        print("\\
=== STATISTICAL SUMMARY ===")
        print(self.df.describe())

        print("\\
=== CATEGORICAL DISTRIBUTIONS ===")
        categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
        for col in categorical_cols:
            print(f"\\
{col}:")
            print(self.df[col].value_counts())

    def clean_data(self):
        """Clean and preprocess the data."""
        if self.df is None:
            print("Please load data first!")
            return

        print("\\
=== DATA CLEANING ===")
        initial_rows = len(self.df)

        # Remove any missing values
        self.df = self.df.dropna()
        print(f"Removed {initial_rows - len(self.df)} rows with missing values")

        # Remove any duplicate rows
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"Removed {initial_rows - len(self.df)} duplicate rows")

        # Convert data types if needed
        self.df['Year'] = self.df['Year'].astype(int)
        self.df['Owner'] = self.df['Owner'].astype(int)

        print(f"Final dataset shape: {self.df.shape}")

    def encode_categorical_features(self):
        """Encode categorical features for machine learning."""
        if self.df is None:
            print("Please load and clean data first!")
            return

        print("\\
=== ENCODING CATEGORICAL FEATURES ===")

        # Initialize label encoders
        categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']

        for feature in categorical_features:
            le = LabelEncoder()
            encoded_col = f"{feature}_encoded"
            self.df[encoded_col] = le.fit_transform(self.df[feature])
            self.label_encoders[feature] = le

            print(f"{feature} encoded:")
            for i, label in enumerate(le.classes_):
                print(f"  {label} -> {i}")

    def prepare_features(self):
        """Prepare features and target for machine learning."""
        if self.df is None:
            print("Please process data first!")
            return None, None

        # Features (X) and target (y)
        X = self.df[self.feature_columns]
        y = self.df['Selling_Price']

        print(f"\\
Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns: {self.feature_columns}")

        return X, y

    def train_model(self, test_size=0.2, random_state=42):
        """Train the Random Forest model for price prediction."""
        X, y = self.prepare_features()
        if X is None:
            return

        print("\\
=== MODEL TRAINING ===")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )

        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\\
Model Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Accuracy: {r2*100:.2f}%")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\\
Feature Importance:")
        print(feature_importance)

        return self.model

    def predict_price(self, car_specs):
        """
        Predict car price based on specifications.

        Args:
            car_specs (dict): Dictionary with car specifications
                - year: int
                - present_price: float
                - driven_kms: int
                - fuel_type: str ('Petrol', 'Diesel', 'CNG')
                - selling_type: str ('Dealer', 'Individual')
                - transmission: str ('Manual', 'Automatic')
                - owner: int

        Returns:
            float: Predicted selling price
        """
        if self.model is None:
            print("Please train the model first!")
            return None

        try:
            # Encode categorical features
            fuel_encoded = self.label_encoders['Fuel_Type'].transform([car_specs['fuel_type']])[0]
            seller_encoded = self.label_encoders['Selling_type'].transform([car_specs['selling_type']])[0]
            transmission_encoded = self.label_encoders['Transmission'].transform([car_specs['transmission']])[0]

            # Prepare feature vector
            features = np.array([[
                car_specs['year'],
                car_specs['present_price'],
                car_specs['driven_kms'],
                fuel_encoded,
                seller_encoded,
                transmission_encoded,
                car_specs['owner']
            ]])

            # Make prediction
            predicted_price = self.model.predict(features)[0]
            return max(0, predicted_price)  # Ensure non-negative price

        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    def save_model(self, model_path='car_price_model.pkl'):
        """Save the trained model and encoders."""
        if self.model is None:
            print("No model to save!")
            return

        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {model_path}")

    def load_model(self, model_path='car_price_model.pkl'):
        """Load a pre-trained model."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']

            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def generate_dashboard_data(self, output_file='dashboard_data.json'):
        """Generate JSON data for the dashboard."""
        if self.df is None:
            print("Please load data first!")
            return

        # Prepare data for dashboard
        dashboard_data = {
            'cars': self.df.to_dict('records'),
            'statistics': {
                'total_cars': len(self.df),
                'avg_selling_price': float(self.df['Selling_Price'].mean()),
                'avg_present_price': float(self.df['Present_Price'].mean()),
                'avg_driven_kms': float(self.df['Driven_kms'].mean()),
                'year_range': {
                    'min': int(self.df['Year'].min()),
                    'max': int(self.df['Year'].max())
                }
            },
            'distributions': {
                'fuel_type': self.df['Fuel_Type'].value_counts().to_dict(),
                'transmission': self.df['Transmission'].value_counts().to_dict(),
                'selling_type': self.df['Selling_type'].value_counts().to_dict()
            }
        }

        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2)

        print(f"Dashboard data saved to {output_file}")
        return dashboard_data


def main():
    """Main function to run the complete data processing pipeline."""
    print("=== CAR MARKET ANALYTICS - DATA PROCESSING ===\\
")

    # Initialize processor
    processor = CarDataProcessor()

    # Load data
    if not processor.load_data():
        return

    # Explore data
    processor.explore_data()

    # Clean data
    processor.clean_data()

    # Encode categorical features
    processor.encode_categorical_features()

    # Train model
    processor.train_model()

    # Save model
    processor.save_model()

    # Generate dashboard data
    processor.generate_dashboard_data()

    print("\\
=== PROCESSING COMPLETE ===")
    print("Files generated:")
    print("- car_price_model.pkl (trained model)")
    print("- dashboard_data.json (dashboard data)")

    # Example prediction
    print("\\
=== EXAMPLE PREDICTION ===")
    example_car = {
        'year': 2015,
        'present_price': 8.5,
        'driven_kms': 35000,
        'fuel_type': 'Petrol',
        'selling_type': 'Dealer',
        'transmission': 'Manual',
        'owner': 0
    }

    predicted_price = processor.predict_price(example_car)
    if predicted_price:
        print(f"Example car specs: {example_car}")
        print(f"Predicted selling price: ₹{predicted_price:.2f} Lakhs")


if __name__ == "__main__":
    main()
'''

# Save the data processing script
with open('data_processing.py', 'w', encoding='utf-8') as f:
    f.write(data_processing_script)

print("Created: data_processing.py")