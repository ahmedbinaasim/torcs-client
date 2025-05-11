import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_and_explore_data():
    """Load and perform basic exploration of the dataset"""
    # Load the CSV file
    data_path = 'cleaned_data.csv'
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Sample of the data:\n{df.head()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values[missing_values > 0]}")
    
    return df

def preprocess_data(df):
    """Preprocess the data for model training"""
    # Remove non-informative columns
    columns_to_drop = ['timestamp', 'source_file']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Handle missing values
    df = df.fillna(0)
    
    # Select only the most essential engineered features for faster training
    # Calculate speed magnitude
    if all(col in df.columns for col in ['speedx', 'speedy', 'speedz']):
        df['speed_magnitude'] = np.sqrt(df['speedx']**2 + df['speedy']**2 + df['speedz']**2)
    
    # Calculate average track sensor value
    track_cols = [col for col in df.columns if col.startswith('track_')]
    if track_cols:
        df['avg_track_distance'] = df[track_cols].mean(axis=1)
    
    # Calculate average opponent distance
    opponent_cols = [col for col in df.columns if col.startswith('opponent_')]
    if opponent_cols:
        df['avg_opponent_distance'] = df[opponent_cols].mean(axis=1)
    
    # Sine and cosine transforms for angle (circular feature)
    if 'angle' in df.columns:
        df['angle_sin'] = np.sin(df['angle'])
        df['angle_cos'] = np.cos(df['angle'])
    
    return df

def train_model(df):
    """Train a single, efficient model"""
    # Define features and target variables
    target_variables = ['accel', 'brake', 'clutch', 'steer']
    feature_columns = [col for col in df.columns if col not in target_variables]
    
    X = df[feature_columns]
    y = df[target_variables]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create a fast-training MLP with good parameters
    print("\nTraining a single MLP model for all controls...")
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),  # Smaller network for faster training
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.002,
        max_iter=200,  # Fewer iterations for faster training
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=True
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nOverall Mean Squared Error: {mse}")
    print(f"Overall R² Score: {r2}")
    
    # For each target variable, display individual metrics
    for i, target in enumerate(target_variables):
        target_mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        target_r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        print(f"{target} - MSE: {target_mse}, R²: {target_r2}")
    
    # Save the model and scaler
    model_path = 'torcs_mlp_model.joblib'
    scaler_path = 'torcs_scaler.joblib'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    return model, scaler, X_test_scaled, y_test, y_pred

def visualize_results(y_test, y_pred, target_variables):
    """Visualize the model's predictions vs actual values"""
    plt.figure(figsize=(15, 10))
    
    for i, target in enumerate(target_variables):
        plt.subplot(2, 2, i+1)
        plt.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.3)
        plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 
                 [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 
                 'r--')
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'{target} Prediction')
    
    plt.tight_layout()
    plt.savefig('model_predictions.png')
    plt.show()

if __name__ == "__main__":
    print("Loading TORCS racing data...")
    df = load_and_explore_data()
    
    print("\nPreprocessing data...")
    processed_df = preprocess_data(df)
    
    print("\nTraining a single model for TORCS racing...")
    model, scaler, X_test_scaled, y_test, y_pred = train_model(processed_df)
    
    print("\nVisualizing results...")
    visualize_results(y_test, y_pred, ['accel', 'brake', 'clutch', 'steer'])
    
    print("\nModel training and evaluation completed!")
