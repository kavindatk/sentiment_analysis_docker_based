#*****************************************************************************************************
#*****************************************************************************************************
#   Author : Kavinda Thennakoon
#   Date : 2024-10-08
#   Period : On-Deamnd
#   Stakeholder : Git Users
#   Program : Sentiment Analysis
#*****************************************************************************************************
#*****************************************************************************************************

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data_files():
    
    #read csv file and remove unwanted columns 
    columns = ['tweet_id','entity','sentiment','tweet_content']
    df = pd.read_csv("../Tweets/twitter_training.csv",names=columns)
    
    df.drop(columns=['tweet_id', 'entity'], inplace=True)
    
    return df


def show_file_details(df):
    
    print('***************Sample Tweets contents***************')
    print(df.head())
    
    
    print('***************[initial] Dataframe Shape***************')
    print(df.shape)

    print('***************[initial] Dataframe Duplicate Check***************')
    print(df.duplicated().sum())
    
    print('***************[initial] Dataframe Null Check***************')
    print(df.isnull().sum())    

    print('***************[initial] Dataframe summary check***************')
    print(df.describe())    

def data_preprocess_stage_01(df_all):
    
    filter_data = ['Positive','Negative']
    
    df = df_all[df_all['sentiment'].isin(filter_data)]
    
    print('***************[Post] Dataframe Duplicate remove check***************')
    df = df.drop_duplicates()
    print(df.duplicated().sum())
    
    print('***************[Post] Dataframe drop null check***************')
    df = df.dropna()
    print(df.isnull().sum()) 
    
    print('***************[Post] Dataframe summary check***************')
    print(df.describe())
    
    return df

    
    
def main():
    tweets = read_data_files()
    
    show_file_details(tweets)
    
    filtered_tweets = data_preprocess_stage_01(tweets)
    

if __name__ == '__main__' :
    main()