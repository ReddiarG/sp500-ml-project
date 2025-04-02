# S&P 500 Stock Removal Prediction 
## Supervised Machine Learning: Binary Classification

This repository contains a Machine Learning project that aims to classify stocks that are likely to be removed from the S&P 500 index using machine learning techniques.

## Project Structure

```
sp500-ml-project/
├── data/
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
├── src/
│   ├── data/         # Data collection scripts
│   ├── features/     # Feature engineering and data preparation scripts
│   └── models/       # Model training and evaluation scripts
├── ml_analysis.ipynb # Main ML analysis notebook
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
└── LICENSE
```

## Setup Instructions

1. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Register the virtual environment as a Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name=sp500-ml --display-name="Python (sp500-ml)"
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

5. Open `ml_analysis.ipynb` and select the "Python (sp500-ml)" kernel from the kernel menu.

## Usage

1. Run the data collection script:
   ```bash
   python src/data/collect.py
   ```

2. Run the feature engineering script:
   ```bash
   python src/features/feature_processor.py
   ```

3. Open and run the ML analysis notebook:
   ```bash
   jupyter notebook ml_analysis.ipynb
   ```

## Project Components

### 1. Data Collection (`src/data_collection/`)
- Contains functions for fetching S&P 500 data
- Note: Due to the nature of financial data sources and potential API changes, the data collection process may not be fully reproducible
- The raw data files are included in the repository to ensure the analysis can be reproduced
- Example usage is provided in the module's docstring

### 2. Feature Engineering and Data Preparation (`src/features/`)
- Contains functions for:
  - Data cleaning and preprocessing
  - Removes tickers with insufficient data
  - Creates derived features and indicators
  - Target variable labelling
- Note: This module is tightly coupled with the data collection process and relies on:
  - Specific file naming conventions from the data collection process
  - Raw data files in CSV format with specific structure
  - Data collection process to generate the correct labels and date ranges
- The feature engineering process is designed to work with the data collection output and may not be directly applicable to different data sources

### 3. Machine Learning Analysis (`notebooks/ml_analysis.ipynb`)
- Main notebook demonstrating the ML workflow
- Requires preprocessed data with features and target variable
- Includes:
  - Implements various ML models
  - Performs hyperparameter tuning
  - Evaluates model performance
  - Visualizes results

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This project is for educational purposes only. The predictions and analysis should not be used for actual trading decisions. 