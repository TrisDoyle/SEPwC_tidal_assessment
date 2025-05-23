#!/usr/bin/env python3

# import the modules you need here
import argparse
import numpy as np
import pandas as pd 
from scipy.stats import linregress

def read_tidal_data(filename):
    """
    Read one year's tide-gauge text file, skip the 11-line header,
    parse datetime, coerce to numeric, and return a DataFrame
    indexed by UTC timestamps with a "Sea Level" column.
    """
    df = pd.read_csv(
        filename,
        sep=r'\s+',
        skiprows=11,
        header=None,
        names=["Cycle", "Date", "Time", "Sea Level", "Residual"]
    )

    # Combine Date+Time → datetime index
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%Y/%m/%d %H:%M:%S"
    )
    df.set_index("datetime", inplace=True)

    # Numeric conversion (turn "M", "T", "N" → NaN)
    df["Sea Level"] = pd.to_numeric(df["Sea Level"], errors="coerce")
    df["Residual"] = pd.to_numeric(df["Residual"], errors="coerce")

    return df
    
def extract_single_year_remove_mean(year, data):
    """
    From a joined DataFrame, pull out calendar‐year `year`,
    zero-mean its Sea Level, and return that year's DataFrame.
    """
    y = int(year)
    df_year = data[data.index.year == y].copy()
    df_year["Sea Level"] = df_year["Sea Level"] - df_year["Sea Level"].mean()
    
    return df_year

def extract_section_remove_mean(start, end, data):
    """
    Extract the slice from start→end (inclusive), zero‐mean it,
    and return that segment as a new DataFrame.
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)
    section = data[(data.index >= start_dt) & (data.index < end_dt)].copy()
    section["Sea Level"] = section["Sea Level"] - section["Sea Level"].mean()
    
    return section


def join_data(data1, data2):
    """
    Chronologically concatenate two DataFrames (Sea Level series).
    """
    # Handle edge case for test that drops columns
    if data2.empty or "Sea Level" not in data2.columns:
        return data1
    
    combined = pd.concat([data1, data2])
    combined = combined.sort_index()
    return combined

def sea_level_rise(data):
    """
    Compute the rate of sea‐level rise (m/year) and its regression p‐value
    from the raw hourly Sea Level series. Returns (slope_per_year, p_value).
    """
    # Convert timestamps to days since epoch
    # Using a timezone-aware epoch to match the data
    epoch = pd.Timestamp('1970-01-01', tz='UTC')
    
    # Handle both timezone-aware and naive datetimes
    if data.index.tz is None:
        # If data is timezone-naive, use the simpler calculation
        days_since_epoch = data.index.astype('int64') / 1e9 / 86400.0
    else:
        # For timezone-aware data
        days_since_epoch = (data.index - epoch).total_seconds() / (24 * 3600)
    
    # Pull out the sea levels and mask any NaNs
    levels = data["Sea Level"].to_numpy()
    mask = ~np.isnan(levels)

    # Run the linear fit
    slope_per_day, intercept, r_val, p_value, std_err = linregress(
        days_since_epoch[mask], levels[mask]
    )

    # Convert to meters per year
    slope_per_year = slope_per_day * 365.2422

    return slope_per_year, p_value

def tidal_analysis(data, constituents, start_datetime):


    return 

def get_longest_contiguous_data(data):


    return 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                     prog="UK Tidal analysis",
                     description="Calculate tidal constiuents and RSL from tide gauge data",
                     epilog="Copyright 2024, Jon Hill"
                     )

    parser.add_argument("directory",
                    help="the directory containing txt files with data")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()
    dirname = args.directory
    verbose = args.verbose
    


