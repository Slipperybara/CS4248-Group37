import requests
import zipfile
import io
import os
import datetime
from collections import defaultdict
from dateutil.relativedelta import relativedelta
import databento as db

import yfinance as yf
import pandas as pd
import numpy as np
import json
import re
from pandas.tseries.offsets import BDay

class Yahoo_Data:

    @classmethod
    def fetch_data(cls, symbol, start_date, end_date):
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    @classmethod
    def process_data(cls, df, market):
        if market == 'currency':
            n_digit = 4
        else:
            n_digit = 2
        df = df.reset_index()
        df[['Open','High','Low','Close']] = df[['Open','High','Low','Close']].apply(lambda x: round(x, n_digit))
        df = df[selected_cols]
        return df

    @classmethod
    def get_data(cls, start_date, end_date, raw_data_directory):
        for row in data_ref:
            if row['data_source'] != 'yahoo':
                continue

            print(f"Fetching data for {row['name']}...")
            data = cls.fetch_data(row['symbol'], start_date, end_date)
            
            if data is not None and not data.empty:
                # Convert the index to string format
                data.index = data.index.strftime('%Y-%m-%d')
                data['Product Name'] = row['name']
                data['Symbol'] = row['symbol']
            
                output_fname = f"{row['name']}.csv"
                output_path = os.path.join(raw_data_directory, output_fname)
                data.to_csv(output_path)
                print(f'{len(data)} rows saved to {output_path}.')
                processed_data = cls.process_data(data, row['market'])
                if os.path.exists(processed_data_output_path):
                    processed_data.to_csv(processed_data_output_path, mode='a', index=False, header=False)
                else:
                    processed_data.to_csv(processed_data_output_path, index=False)

class CME_Data:

    month_code_map = {
            "January": "F", "February": "G", "March": "H", "April": "J",
            "May": "K", "June": "M", "July": "N", "August": "Q",
            "September": "U", "October": "V", "November": "X", "December": "Z"
        }

    market_config = {
        "Live Cattle Future": {
            "symbol": "LE",
            "months": ["February", "April", "June", "August", "October", "December"]
        },
        "Feeder Cattle Future": {
            "symbol": "GF",
            "months": ["January", "March", "April", "May", "August", "September", "October", "November"]
        },
        "Lean Hogs Future": {
            "symbol": "HE",
            "months": ["February", "April", "May", "June", "July", "August", "October", "December"]
        },
        "Corn Future": {
            "symbol": "ZC",
            "months": ["March", "May", "July", "September", "December"]
        },
        "Chicago SRW Wheat Future": {
            "symbol": "ZW",
            "months": ["March", "May", "July", "September", "December"]
        },
        "KC HRW Wheat Future": {
            "symbol": "KE",
            "months": ["March", "May", "July", "September", "December"]
        },
        "Class III Milk Future": {
            "symbol": "DC",
            "months": list(month_code_map.keys())  # All months
        },
        "Class IV Milk Future": {
            "symbol": "DK",
            "months": list(month_code_map.keys())  # All months
        },
        "Soybeans Future": {
            "symbol": "ZS",
            "months": ["January", "March", "May", "July", "August", "September", "November"]
        },
        "Soybean Oil Future": {
            "symbol": "ZL",
            "months": ["January", "March", "May", "July", "August", "September", "October", "December"]
        },
        "Soybean Meal Future": {
            "symbol": "ZM",
            "months": ["January", "March", "May", "July", "August", "September", "October", "December"]
        }
    }

    @classmethod
    def get_contract_months(cls, product):
        """Get the valid contract months for a given market"""

        # Check if market exists in config
        if product in cls.market_config:
            return cls.market_config[product]["months"]
        
        # For any other market not specifically configured
        print(f"Warning: No specific configuration for '{product}'. Using all months.")
        return list(cls.month_code_map.keys())

    @classmethod
    def get_last_trading_day(cls, year, month, market):
        """Calculate the last trading day based on market specifications"""
        # Create date for first day of next month
        if month == 12:
            next_month = datetime.datetime(year + 1, 1, 1)
        else:
            next_month = datetime.datetime(year, month + 1, 1)
        
        # Get last day of current month
        last_day = next_month - datetime.timedelta(days=1)
        
        # Adjust based on market specifications
        market = market.lower()
        if market in ["cattle", "lean hog"]:
            # Last business day of the month
            while last_day.weekday() in [5, 6]:  # Saturday = 5, Sunday = 6
                last_day = last_day - datetime.timedelta(days=1)
        elif market in ["corn", "wheat", "soybean"]:
            # 15th calendar day of the contract month
            last_day = datetime.datetime(year, month, 15)
        elif market in ['dairy']:
            # Last trading day is one business day before the fifth business day of the following month --> fourth business day of the following month
            last_day = next_month + BDay(4)

        return last_day

    @classmethod
    def generate_futures_symbols(cls, start_year, end_year):
        symbols = []

        for row in data_ref:
            if row['data_source'] != 'cme' or not row["symbol"]:
                continue

            product = row["name"]
            contract_months = cls.get_contract_months(product)

            for year in range(start_year, end_year + 1):
                for month in contract_months:
                    month_num = list(cls.month_code_map.keys()).index(month) + 1
                    
                    # Calculate expiration date
                    expiration_date = cls.get_last_trading_day(year, month_num, row['market'])

                    month_code = cls.month_code_map[month]
                    year_code = str(year)[-1]
                    full_symbol = f"{row['symbol']}{month_code}{year_code}"

                    symbols.append({
                        "market": row["market"],
                        "name": row["name"],
                        "symbol": full_symbol,
                        "expiration_date": expiration_date.strftime("%Y-%m-%d")
                    })

        df_symbols = pd.DataFrame(symbols)
        df_symbols.to_csv('data/references/futures_symbol.csv', index=False)
        return df_symbols

    @classmethod
    def format_datetime(cls, orig_date):
        date_str = orig_date.strftime("%Y-%m-%dT%H:%M:%S")
        return date_str

    @classmethod
    def process_data(cls, df):
        df = df.reset_index()
        df = df.rename(columns={'ts_event': 'Date', 'open': 'Open','high': 'High','low': 'Low','close': 'Close','volume': 'Volume','symbol': 'Symbol', 'product name': 'Product Name'},)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df[selected_cols]
        return df

    @classmethod
    def get_data(cls, start_date, end_date, raw_data_directory):

        symbol_end_date = end_date + datetime.timedelta(days=365) 
        df_symbols = cls.generate_futures_symbols(start_date.year, symbol_end_date.year)
        grouped_symbols = df_symbols.groupby('name')['symbol'].apply(list)
        symbols_dict = grouped_symbols.to_dict()

        # Offset by 2 years
        data_start_date = start_date - datetime.timedelta(days=2*365) 
        data_start_date_str = cls.format_datetime(data_start_date)
        data_end_date_str = cls.format_datetime(end_date)

        client = db.Historical('db-Aev8CxUWsE37Vr6rGQc8gPxchVLfm') # Include your own api key.

        for future, future_symbols in symbols_dict.items():
            print(f'Downloading {future} data...')
            data = client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=future_symbols,
                schema="ohlcv-1d",
                start=data_start_date_str,
                end=data_end_date_str,
            )
            df = data.to_df()
            df['product name'] = future
            output_fname = f"{future}.csv"
            output_path = os.path.join(raw_data_directory, output_fname)
            df.to_csv(output_path)
            print(f'{len(df)} rows saved to {output_path}.')
            processed_df = cls.process_data(df)
            processed_df.to_csv(processed_data_output_path, mode='a', index=False, header=False)

class LH_data:

    @classmethod
    def download_and_unzip(cls, url, extract_to):
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Successfully downloaded and extracted to: {extract_to}")
        else:
            print(f"Failed to download: {url}, please download and unzip to {extract_to} lhindex.xls manually, then rerun the ")

    @classmethod
    def process_data(cls, df):
        def process_date(date_object):
            if type(date_object) == datetime.datetime:
                return date_object.strftime("%Y-%m-%d")
            else:
                return ""
        
        df['Date'] = df['Date'].apply(process_date)
        df['Close'] = df['CME INDEX'].apply(lambda x:round(x, 2))
        df['Product Name'] = 'Lean Hog Cash Index'
        df['Symbol'] = ''
        for col in selected_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[selected_cols]
        return df

    @classmethod
    def get_data(cls, raw_data_directory):
        lean_hog_url = "https://www.cmegroup.com/ftp/cash_settled_commodity_index_prices/historical_data/LHindx.ZIP"
        cls.download_and_unzip(lean_hog_url, 'data/raw/cme')

        # Rename the extracted file to cme_lean_hog.xls
        extracted_files = os.listdir(raw_data_directory)
        for filename in extracted_files:
            if filename.endswith('.xls'):
                os.rename(os.path.join(raw_data_directory, filename), os.path.join(raw_data_directory, 'lhindx.xls'))
                break

        df = pd.read_excel(os.path.join(raw_data_directory, 'lhindx.xls'), header=2)

        processed_df = cls.process_data(df)
        processed_df.to_csv(processed_data_output_path, mode='a', index=False, header=False)


class FC_Data:
    @classmethod
    def process_data(cls, df):
        def process_date(date_string):
            try:
                # Convert string date to datetime object then format it
                date_object = datetime.strptime(date_string, "%m/%d/%Y")
                return date_object.strftime("%Y-%m-%d")
            except:
                return ""
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'Date': 'Date',
            'Value 1': 'Close'
        })
        
        # Process date column
        df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
        
        # Round values to 2 decimal places
        df['Close'] = df['Close'].apply(lambda x: round(float(x), 2) if pd.notnull(x) else np.nan)
        
        # Add required columns
        df['Product Name'] = 'Feeder Cattle Cash Index'
        df['Symbol'] = ''
        df['Volume'] = np.nan
        
        # Ensure all required columns are present
        for col in selected_cols:
            if col not in df.columns:
                df[col] = np.nan
                
        df = df[selected_cols]
        
        return df

    @classmethod
    def get_data(cls, raw_data_directory):
        file_path = os.path.join(raw_data_directory, 'feeder_cattle_index.csv')
        df = pd.read_csv(file_path)
        
        processed_df = cls.process_data(df)
        processed_df.to_csv(processed_data_output_path, mode='a', index=False, header=False)

class WSJ_Data:

    @classmethod
    def match_file_to_bond(cls, filename):
        # Extract the year from the filename
        match = re.search(r'us_(\d+)_year_bond_yield\.csv', filename)
        if match:
            year = match.group(1)
            for bond in data_ref:
                if f"{year}-Year" in bond["name"]:
                    bond['maturity'] = int(year)
                    return bond
        return None
    
    @classmethod
    def process_date(cls, date_string):
        date_object = datetime.strptime(date_string, "%m/%d/%y")
        formatted_date = date_object.strftime("%Y-%m-%d")
        return formatted_date

    @classmethod
    def process_data(cls, raw_data_directory):
        for filename in os.listdir(raw_data_directory):
            if filename.endswith('.csv'):
                bond = cls.match_file_to_bond(filename)
                if bond:
                    filepath = os.path.join(raw_data_directory, filename)
                    df = pd.read_csv(filepath)
                    df.rename(columns={x:x.strip() for x in df.columns}, inplace=True)
                    df['Product Name'] = bond['name']
                    df['Symbol'] = f'TMUBMUSD{bond["maturity"]:02d}Y'
                    df['Volume'] = np.nan
                    if pd.api.types.is_string_dtype(df['Date']) or pd.api.types.is_object_dtype(df['Date']):
                        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y').dt.strftime("%Y-%m-%d")
                    df = df[selected_cols]
                    df.to_csv(processed_data_output_path, mode='a', index=False, header=False)


def main():
    # 1. Download equity, energy and FX data
    yahoo_data_dir = os.path.join(raw_data_dir, 'yahoo_finance')
    os.makedirs(yahoo_data_dir, exist_ok=True)
    equity_data_start_date =  datetime.date(2018, 6, 1)
    equity_data_end_date = datetime.date(2023, 3, 31)
    Yahoo_Data.get_data(equity_data_start_date, equity_data_end_date, raw_data_directory=yahoo_data_dir)

    # 2. Download agriculture commodity data
    cme_data_dir = os.path.join(raw_data_dir, 'cme')
    os.makedirs(cme_data_dir, exist_ok=True)
    agriculture_report_start_date = datetime.date(2021, 6, 1)
    agriculture_report_end_date = datetime.date(2023, 4, 30)
    CME_Data.get_data(agriculture_report_start_date, agriculture_report_end_date, raw_data_directory=cme_data_dir)
    # LH_data.get_data(raw_data_directory=cme_data_dir)
    # FC_Data.get_data(raw_data_directory=cme_data_dir)

    # 3. Process treasury data
    wsj_data_dir = os.path.join(raw_data_dir, 'wsj')
    WSJ_Data.process_data(raw_data_directory=wsj_data_dir)

if __name__ == "__main__":
    with open('data/references/financial_instrument_reference.json', 'r') as f:
        data_ref = json.load(f)

    # Create data/raw directory if it doesn't exist
    table_data_dir = os.path.join('data', 'tabular_data')
    raw_data_dir = os.path.join(table_data_dir, 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)

    processed_data_output_path = os.path.join(table_data_dir, 'intermediate', 'processed_data.csv')
    selected_cols = ['Date', 'Product Name', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
    main()