"""
Data collection module for S&P 500 stock data.

This module handles the collection of historical stock data for S&P 500 companies.
It processes a CSV file containing ticker symbols and their respective date ranges,
downloads the data using yfinance, and saves it to CSV files.

Note: Due to the nature of financial data sources and potential API changes,
the data collection process may not be fully reproducible. The raw data files
are included in the repository to ensure the analysis can be reproduced.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SP500DataFetcher:
    """A class to handle S&P 500 data collection and processing."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the SP500DataFetcher.
        
        Args:
            data_dir (Union[str, Path]): Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / 'raw' / 'ticker_data'
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_ticker_dates(self, csv_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load and process the S&P 500 ticker dates from CSV.
        
        Args:
            csv_path (Union[str, Path]): Path to the CSV file containing ticker dates
            
        Returns:
            pd.DataFrame: Processed DataFrame with ticker dates
        """
        try:
            df = pd.read_csv(csv_path)
            df['start_date'] = pd.to_datetime(df['start_date']).dt.date
            df['end_date'] = pd.to_datetime(df['end_date']).dt.date
            df = df.sort_values(by=['ticker', 'start_date']).reset_index(drop=True)
            
            # Add suffix to duplicate tickers
            df['ticker'] = (
                df.groupby('ticker').cumcount().add(1).astype(str).radd(df['ticker'] + '_')
                .where(df.groupby('ticker').cumcount() > 0, df['ticker'])
            )
            
            return df.set_index('ticker')
        except Exception as e:
            logger.error(f"Error loading ticker dates: {e}")
            raise
            
    def download_ticker_data(self, ticker: str, date_range: Dict) -> Optional[pd.DataFrame]:
        """
        Download data for a specific ticker within its date range.
        
        Args:
            ticker (str): The ticker symbol
            date_range (Dict): Dictionary containing start_date and end_date
            
        Returns:
            Optional[pd.DataFrame]: Downloaded data or None if download fails
        """
        try:
            ticker_only = ticker.strip('_')[0]
            start = date_range.get('start_date')
            end = date_range.get('end_date')
            
            if pd.isna(end):
                end = None
                
            data = yf.download(ticker_only, start=start, end=end)
            data['Ticker'] = ticker
            data['Time Frame'] = f"{start} to {end}"
            
            return data
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {e}")
            return None
            
    def save_ticker_data(self, data: pd.DataFrame, ticker: str, date_range: Dict) -> None:
        """
        Save ticker data to CSV file.
        
        Args:
            data (pd.DataFrame): The data to save
            ticker (str): The ticker symbol
            date_range (Dict): Dictionary containing start_date and end_date
        """
        try:
            start = date_range.get('start_date')
            end = date_range.get('end_date')
            filename = f"{ticker}_prices_{start}_to_{end}.csv"
            filepath = self.raw_data_dir / filename
            data.to_csv(filepath)
            logger.info(f"Saved data for {ticker} to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data for {ticker}: {e}")
            
    def fetch_all_tickers(self, csv_path: Union[str, Path]) -> None:
        """
        Fetch data for all tickers in the CSV file.
        
        Args:
            csv_path (Union[str, Path]): Path to the CSV file containing ticker dates
        """
        try:
            sp_all_time = self.load_ticker_dates(csv_path).to_dict('index')
            
            for ticker, date_range in sp_all_time.items():
                logger.info(f"Processing {ticker}")
                data = self.download_ticker_data(ticker, date_range)
                if data is not None:
                    self.save_ticker_data(data, ticker, date_range)
                    
        except Exception as e:
            logger.error(f"Error in fetch_all_tickers: {e}")
            raise

def main():
    """Example usage of the SP500DataFetcher class."""
    # Initialize the fetcher
    data_dir = Path(__file__).parent.parent.parent / 'data'
    fetcher = SP500DataFetcher(data_dir)
    
    # Path to the ticker dates CSV
    csv_path = data_dir / 'sp500_ticker_start_end.csv'
    
    # Fetch all ticker data
    fetcher.fetch_all_tickers(csv_path)

if __name__ == "__main__":
    main()
