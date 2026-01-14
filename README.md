# AutoGluon Library Implementation

This repository demonstrates **AutoGluon** - a powerful AutoML library that automates machine learning tasks with minimal code.

## 🚀 Features

- **Classification Example**: Titanic survival prediction (binary classification)
- **Interactive Tutorial**: Comprehensive Jupyter notebook with multiple examples
- **Regression & Classification**: Both problem types covered
- **Kaggle Integration**: Work with real datasets
- **Feature Importance**: Automatic feature importance analysis
- **Model Leaderboard**: Compare multiple models automatically

## 📋 Prerequisites

- Python 3.8+
- pip package manager

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yadavanujkumar/autogluon-library-test.git
cd autogluon-library-test
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: AutoGluon installation may take 5-10 minutes as it includes multiple ML frameworks.

## 🎯 Quick Start

### Option 1: Run the Classification Example (Fastest)
```bash
python autogluon_classification.py
```

**What it does**:
- Uses the Titanic dataset (built-in, no download needed)
- Trains 10+ different ML models automatically
- Shows model leaderboard and performance metrics
- Generates feature importance visualization
- Makes sample predictions

**Expected runtime**: ~3-5 minutes

### Option 2: Interactive Jupyter Notebook (Recommended for Learning)
```bash
jupyter notebook autogluon_tutorial.ipynb
```

**What's included**:
- Step-by-step tutorial with explanations
- Classification example (Titanic dataset)
- Regression example (California Housing)
- Kaggle dataset integration
- Visualization and interpretation
- Customization examples

## 📊 Expected Output

When you run the examples, you'll get:
- **Model Files**: Saved in `./autogluon_models/`
- **Visualizations**: Feature importance plots (PNG files)
- **Performance Metrics**: Accuracy, R², RMSE, etc.
- **Leaderboard**: Comparison of all trained models

Example output:
```
Training AutoGluon Model
========================
Training 10+ models...
✓ RandomForest
✓ XGBoost  
✓ LightGBM
✓ CatBoost
✓ Neural Networks
✓ Ensemble models

Best model: WeightedEnsemble_L2
Test Accuracy: 82.12%
```

## 📖 AutoGluon Key Features

1. **AutoML**: Automatically tries multiple models and ensembles
2. **Minimal Code**: Train state-of-art models with 3 lines of code
3. **Feature Engineering**: Automatic feature preprocessing
4. **Hyperparameter Tuning**: Automatic optimization
5. **Model Stacking**: Combines multiple models for better performance
6. **Multi-modal**: Supports tabular, text, and image data

## 🛠️ Customization

### Adjust Training Time
```python
predictor.fit(train_data, time_limit=300)  # 5 minutes
```

### Change Quality Presets
```python
# Options: 'best_quality', 'high_quality', 'good_quality', 'medium_quality'
predictor.fit(train_data, presets='best_quality')
```

### Specify Models
```python
predictor.fit(
    train_data,
    hyperparameters={
        'GBM': {},      # LightGBM
        'XGB': {},      # XGBoost
        'CAT': {},      # CatBoost
        'RF': {},       # Random Forest
        'NN_TORCH': {}  # Neural Network
    }
)
```

## 📁 Project Structure

```
autogluon-library-test/
├── autogluon_classification.py    # Quick classification example (Titanic)
├── autogluon_tutorial.ipynb       # Complete interactive tutorial
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
└── LICENSE                        # License file
```

**Generated during execution**:
```Troubleshooting

### Module not found error
```bash
pip install -r requirements.txt
```

### Out of Memory
Reduce `time_limit` or use `presets='medium_quality'` in the script

### Training is slow
This is normal! AutoGluon trains many models. For faster results:
- Reduce `time_limit` parameter (e.g., `time_limit=60`)
- Use `presets='medium_quality'` instead of `'best_quality'`

### Kaggle API not configured (in notebook)
The notebook includes fallback to built-in datasets, so it will work without Kaggle setup
### Issue: Slow Training
**Solution**: Start with 'medium_quality' preset or reduce time_limit

### Issue: Kaggle API Error
**Solution**: Ensure kaggle.json is properly configured at ~/.kaggle/kaggle.json

## 📚 Additional Resources

- [AutoGluon Documentation](https://auto.gluon.ai/)
- [AutoGluon GitHub](https://github.com/autogluon/autogluon)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [AutoGluon Tutorials](https://auto.gluon.ai/stable/tutorials/index.html)

## 🤝 Contributing

Feel free to open issues or submit pull requests with improvements!

## 📄 License

See LICENSE file for details.

## 💡 Tips for Best Results

1. **More Time = Better Models**: Increase `time_limit` for better performance
2. **Beginners**: Start with [autogluon_classification.py](autogluon_classification.py) - Run and see results in 3 minutes
2. **Deep Dive**: Open [autogluon_tutorial.ipynb](autogluon_tutorial.ipynb) - Learn interactively with explanations
3. **Advanced**: Modify scripts, try your own datasets, adjust parameters

## 📝 What's in the Notebook?

The [autogluon_tutorial.ipynb](autogluon_tutorial.ipynb) covers:
- ✅ Installation and setup
- ✅ Loading datasets (Kaggle & built-in)
- ✅ Data exploration and preprocessing
- ✅ Classification example (Titanic)
- ✅ Regression example (Housing prices)
- ✅ Model evaluation and metrics
- ✅ Feature importance analysis
- ✅ Making predictions
- ✅ Customization tips


## 🎓 Learning Path

1. Start with `autogluon_classification.py` - Simplest example
2. Try `autogluon_regression.py` - Learn regression tasks
3. Explore `autogluon_kaggle_example.py` - Work with real datasets
4. Deep dive with `autogluon_tutorial.ipynb` - Interactive learning

