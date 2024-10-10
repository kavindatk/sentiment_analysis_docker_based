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
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import StringType,StructType,StructField
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def read_data_files():
    schema = StructType([
        StructField("tweet_id", StringType(), True),
        StructField("entity_id", StringType(), True),
        StructField("setnment_id", StringType(), True),
        StructField("tweet_msg", StringType(), True)
    ])    
    df = spark.read.csv("../Tweets/twitter_training.csv", header=False, schema=schema)
    #df.printSchema()
    #df.show()
    #df.summary().show()
    
    return df
    
def data_pre_process(data_df):
    
    #Filter non null data and useful columns from data set
    df = data_df.filter(
        (f.col("setnment_id").isin('Positive','Negative')) & (f.length(f.trim(f.col("tweet_msg")))>0)
    ).select(
        f.when(f.col("setnment_id")=='Positive',f.lit(1)).otherwise(0).alias('setnment_id'),
        f.col("tweet_msg")
    )
    
    # Tokenize the tweet message
    tokenizer = Tokenizer(inputCol="tweet_msg", outputCol="words")
    df_words = tokenizer.transform(df)

    # Remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    df_clean = remover.transform(df_words)
    
    # Convert words to term frequency vectors
    vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")
    df_tf = vectorizer.fit(df_clean).transform(df_clean)
    
    
    # Apply IDF to get the final feature vectors - TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text data into numerical features.
    idf = IDF(inputCol="raw_features", outputCol="features")
    df_features = idf.fit(df_tf).transform(df_tf)
    
    # Initialize the logistic regression model
    lr = LogisticRegression(labelCol="setnment_id", featuresCol="features") 
    
    # Split the data into training and test sets
    train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=1234)  
    
    # Train the model
    lr_model = lr.fit(train_data)   

    # Make predictions on the test data
    predictions = lr_model.transform(test_data)
    
    
    # Initialize evaluator for classification
    evaluator = MulticlassClassificationEvaluator(
        labelCol="setnment_id", 
        predictionCol="prediction", 
        metricName="accuracy"
    )   
    
    # Calculate accuracy
    accuracy = evaluator.evaluate(predictions)
    print('**************************************************')
    print("Test Accuracy = {}".format(accuracy))         
    print('**************************************************')
    
    return lr_model

    
def main():
    df = read_data_files()
    data_pre_process(df)
    spark.stop()
       

if __name__ == '__main__' :
    spark = SparkSession.builder.master("local[*]").appName("sentiment_analysis").enableHiveSupport().getOrCreate()
    main()