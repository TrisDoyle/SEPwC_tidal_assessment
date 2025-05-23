#!/usr/bin/env python3

# import the modules you need here
import argparse

import pandas as pd 

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
   

    return 


def extract_section_remove_mean(start, end, data):


    return 


def join_data(data1, data2):

    return 



def sea_level_rise(data):

                                                     
    return 

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
    


