"""
Feature Engineering and Data Processing Module for S&P 500 Data.

This module processes raw stock data to create features and labels for machine learning.
It handles:
- Data cleaning and preprocessing
- Feature calculation (price changes, volatility, etc.)
- Label generation
- Data aggregation

Note: This module is tightly coupled with the data collection process and relies on:
- Specific file naming conventions from the data collection process
- Raw data files in CSV format with specific structure (Date, Adj Close, Volume columns)
- Data collection process to generate the correct labels and date ranges

The feature engineering process is designed to work with the data collection output
and may not be directly applicable to different data sources or collection methods.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SP500FeatureProcessor:
    """A class to process S&P 500 stock data and create features for ML."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the SP500FeatureProcessor.
        
        Args:
            data_dir (Union[str, Path]): Path to the directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.processed_data_dir = self.data_dir.parent.parent / 'processed'
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def _calculate_percent_change(self, start: float, end: float) -> float:
        """
        Calculate percentage change between two values.
        
        Args:
            start (float): Starting value
            end (float): Ending value
            
        Returns:
            float: Percentage change or NaN if start is 0
        """
        if start == 0:
            return np.nan
        return ((end - start) / start) * 100
        
    def _process_single_file(self, file_path: Path) -> Optional[Dict]:
        """
        Process a single stock data file and extract features.
        
        Args:
            file_path (Path): Path to the CSV file
            
        Returns:
            Optional[Dict]: Dictionary of features or None if processing fails
        """
        try:
            # Extract information from filename
            file_name = file_path.name
            ticker = file_name.split("_prices_")[0]
            end_date = file_name.split("_to_")[1].split(".")[0]
            label = 0 if end_date == "None" else 1
            
            # Load and preprocess data
            df = pd.read_csv(file_path, skiprows=[1, 2])
            df.rename(columns={'Price': 'Date'}, inplace=True)
            
            # Clean and prepare data
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
            df.sort_values(by='Date', inplace=True)
            
            if df.empty or len(df) < 90:
                logger.warning(f"Skipping {file_name}: insufficient data")
                return None
                
            df["Date"] = pd.to_datetime(df["Date"])
            df[["Adj Close", "Volume"]] = df[["Adj Close", "Volume"]].apply(pd.to_numeric)
            
            # Calculate features
            record = {
                "Ticker": ticker,
                "Latest_Price": df["Adj Close"].iloc[-1],
                "Latest_Volume": df["Volume"].iloc[-1],
                "Latest_Est_Market_Cap": df["Adj Close"].iloc[-1] * df["Volume"].iloc[-1],
                "Total_Return": (df["Adj Close"].iloc[-1] - df["Adj Close"].iloc[0]) / df["Adj Close"].iloc[0],
                "CAGR": (pow((df["Adj Close"].iloc[-1] / df["Adj Close"].iloc[0]), 1 / (len(df) / 252)) - 1) * 100,
                "Label": label,
                "End_date": end_date
            }
            
            # Calculate period-specific features
            for period, days in [("Week", 5), ("Month", 21), ("3Months", 63)]:
                subset = df.tail(days) if len(df) >= days else df
                record.update({
                    f"Last_{period}_Price_Change": (subset["Adj Close"].iloc[-1] - subset["Adj Close"].iloc[0]) / subset["Adj Close"].iloc[0],
                    f"Last_{period}_Average": subset["Adj Close"].mean(),
                    f"Last_{period}_Volatility": subset["Adj Close"].pct_change().std()
                })
            
            return record
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
            
    def process_all_files(self) -> pd.DataFrame:
        """
        Process all stock data files and create a consolidated feature set.
        
        Returns:
            pd.DataFrame: Consolidated DataFrame with all features
        """
        try:
            aggregated_df = pd.DataFrame()
            
            for file in self.data_dir.glob("*.csv"):
                logger.info(f"Processing {file.name}")
                record = self._process_single_file(file)
                if record:
                    aggregated_df = pd.concat([aggregated_df, pd.DataFrame([record])], ignore_index=True)
            
            # Save processed data
            output_path = self.processed_data_dir / "aggregated_stock_features.csv"
            aggregated_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            
            return aggregated_df
            
        except Exception as e:
            logger.error(f"Error in process_all_files: {e}")
            raise

def main():
    """Example usage of the SP500FeatureProcessor class."""
    # Initialize the processor
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'ticker_data'
    processor = SP500FeatureProcessor(data_dir)
    
    # Process all files
    features_df = processor.process_all_files()
    print(f"Processed {len(features_df)} stock records")

if __name__ == "__main__":
    main() 