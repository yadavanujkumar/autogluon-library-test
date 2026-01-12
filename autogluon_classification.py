"""
AutoGluon Classification Example with Titanic Dataset
This script demonstrates how to use AutoGluon for binary classification
"""

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def download_titanic_data():
    """Download and prepare Titanic dataset"""
    print("Loading Titanic dataset...")
    # Using seaborn's built-in titanic dataset
    df = sns.load_dataset('titanic')
    return df


def preprocess_data(df):
    """Basic preprocessing for the dataset"""
    print("\nPreprocessing data...")
    
    # Select relevant features
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']
    df_clean = df[features].copy()
    
    # Drop rows with missing target
    df_clean = df_clean.dropna(subset=['survived'])
    
    print(f"Dataset shape: {df_clean.shape}")
    print(f"\nMissing values:\n{df_clean.isnull().sum()}")
    print(f"\nTarget distribution:\n{df_clean['survived'].value_counts()}")
    
    return df_clean


def train_autogluon_model(train_data, label='survived', time_limit=120):
    """Train AutoGluon model"""
    print(f"\n{'='*60}")
    print("Training AutoGluon Model")
    print(f"{'='*60}")
    
    # Initialize predictor
    predictor = TabularPredictor(
        label=label,
        problem_type='binary',
        eval_metric='accuracy',
        path='./autogluon_models/titanic'
    )
    
    # Train the model
    predictor.fit(
        train_data=train_data,
        time_limit=time_limit,
        presets='best_quality',  # Options: 'best_quality', 'high_quality', 'good_quality', 'medium_quality'
        verbosity=2
    )
    
    return predictor


def evaluate_model(predictor, test_data, label='survived'):
    """Evaluate the trained model"""
    print(f"\n{'='*60}")
    print("Model Evaluation")
    print(f"{'='*60}")
    
    # Get predictions
    y_pred = predictor.predict(test_data.drop(columns=[label]))
    y_true = test_data[label]
    
    # Get prediction probabilities
    y_pred_proba = predictor.predict_proba(test_data.drop(columns=[label]))
    
    # Evaluate performance
    performance = predictor.evaluate(test_data, silent=False)
    
    print(f"\nTest Accuracy: {performance:.4f}")
    
    # Feature importance
    print("\nFeature Importance:")
    feature_importance = predictor.feature_importance(test_data)
    print(feature_importance)
    
    # Model leaderboard
    print("\nModel Leaderboard:")
    leaderboard = predictor.leaderboard(test_data, silent=True)
    print(leaderboard)
    
    return y_pred, y_pred_proba, performance


def visualize_results(predictor, test_data, label='survived'):
    """Visualize model results"""
    feature_importance = predictor.feature_importance(test_data)
    
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='barh')
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nFeature importance plot saved as 'feature_importance.png'")


def main():
    """Main execution function"""
    print("="*60)
    print("AutoGluon Classification - Titanic Survival Prediction")
    print("="*60)
    
    # 1. Load data
    df = download_titanic_data()
    
    # 2. Preprocess
    df_clean = preprocess_data(df)
    
    # 3. Split data
    print("\nSplitting data into train and test sets...")
    train_data, test_data = train_test_split(
        df_clean, 
        test_size=0.2, 
        random_state=42,
        stratify=df_clean['survived']
    )
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # 4. Train model
    predictor = train_autogluon_model(train_data, time_limit=120)
    
    # 5. Evaluate model
    y_pred, y_pred_proba, performance = evaluate_model(predictor, test_data)
    
    # 6. Visualize results
    visualize_results(predictor, test_data)
    
    # 7. Make predictions on new data
    print(f"\n{'='*60}")
    print("Example Predictions")
    print(f"{'='*60}")
    
    # Create sample data for prediction
    sample_data = pd.DataFrame({
        'pclass': [3, 1, 2],
        'sex': ['male', 'female', 'male'],
        'age': [22, 38, 26],
        'sibsp': [1, 1, 0],
        'parch': [0, 0, 0],
        'fare': [7.25, 71.28, 13.0],
        'embarked': ['S', 'C', 'S']
    })
    
    predictions = predictor.predict(sample_data)
    probabilities = predictor.predict_proba(sample_data)
    
    print("\nSample Predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities.values)):
        print(f"\nPassenger {i+1}:")
        print(f"  Prediction: {'Survived' if pred == 1 else 'Did not survive'}")
        print(f"  Confidence: {max(prob):.2%}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Model saved at: ./autogluon_models/titanic")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
