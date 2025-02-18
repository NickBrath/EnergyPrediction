import pandas as pd
import numpy as np

# Feature engineering functions
def get_daylight_times(df:pd.DataFrame) -> pd.Series:
    """
    Returns the range of daylight times for a fleet over many months
    
    Daylight can begin and end at a different times throughout the year for different sites. We assume daylight is any time when the solar production is non-zero. 
    We use AC Power as a proxy variable for solar production. If the sun is in the sky, then we assume that solar production is occuring. Each day might have a 
    different daylight start time, so to generalize the model, we look across all days in this dataframe for earliest and latest daylight times and use this as our range.
    
    Assumptions:
        daylight begins when solar production starts
        daylight ends when solar production finishes for the day
        
    output:
        an ordered series of the times when AC Power was non zero for any site, on any day
        
    """
    
    # times when AC power is non zero
    sunlight_filter = df.groupby('TIME').sum()['AC_POWER'] > 0

    # all non zero times when production occurs
    daylight_times = df.groupby('TIME').sum()['AC_POWER'][sunlight_filter].index
    
    
    return daylight_times

def create_is_daytime_feature(df:pd.DataFrame, sunlight_times:pd.Series=None) -> pd.DataFrame:
    """
    Creates a new feature for the dataframe called is_daylight. is_daylight is a binary variable that represents if the record occured during a time when the sun could be in the sky and 
    solar production could occur. When is_daylight is 0, the probability that the sun is in the sky and solar production can occur is 0. When is_daylight is 1, the sun is up and solar production can occur, but
    is not guaranteed.
    
    output:
        a dataframe with the new feature
    """
    if sunlight_times is None:
        sunlight_times = get_daylight_times(df)
    
    df['is_daytime'] = df.TIME.isin(sunlight_times).astype(int)
    
    return df


def create_yesterday_max_yield(df:pd.DataFrame) -> pd.DataFrame:
    """
    Adds each inverter's previous day's max production value for daily yield as a new column to a dataframe.
    
    If this was the first date for that inverter, it enters null for that record. The value for yesterdays highest production is repeated for all intervals in the day.
    
    input: the dataframe that needs the new feature
    
    output: the same dataframe now with the additional column
    """
    
    # check to see if the field already exists
    if 'yesterday_max_daily_yield' in df.columns:
        print('yesterday_max_daily_yield column already exists. dropping column...')
        df.drop(columns='yesterday_max_daily_yield', inplace=True)
    
    print('creating yesterday_max_daily_yield column ...')
    
    # create the max yield for each inverter-day
    max_yield = df.groupby(["SOURCE_KEY","DATE"])["DAILY_YIELD"].max().reset_index()
    max_yield = max_yield.sort_values(by='DATE')
    max_yield['MAX_YIELD'] = max_yield["DAILY_YIELD"]

    # create a field for yesterday's max yield
    shifted_max_yield = max_yield
    shifted_max_yield["yesterday_max_daily_yield"] = max_yield.groupby(["SOURCE_KEY"])["MAX_YIELD"].shift(1)
    
    # keep only critical columns
    shifted_max_yield = shifted_max_yield[["SOURCE_KEY","DATE","yesterday_max_daily_yield"]]
    
    # merge the yesterday max yield column into the original dataframe. 
    # requires reseting index into a date time column and then setting the index back again
    # otherwise, we lose our datetime index forever
    df = df.reset_index().merge(shifted_max_yield, how="left", on=['SOURCE_KEY','DATE']).set_index('DATE_TIME')
    
    return df

def load_train_test_data():
    train_df = pd.read_csv("processed_data/train_df.csv")
    train_df['DATE_TIME']= pd.to_datetime(train_df['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')
    train_df.drop(columns='Unnamed: 0', inplace=True)


    test_df = pd.read_csv("processed_data/test_df.csv")
    test_df['DATE_TIME']= pd.to_datetime(test_df['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')
    test_df.drop(columns='Unnamed: 0', inplace=True)
    
    return train_df, test_df