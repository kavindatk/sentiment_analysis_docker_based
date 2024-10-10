#*****************************************************************************************************
#*****************************************************************************************************
#   Author : Kavinda Thennakoon
#   Date : 2024-10-08
#   Period : On-Demand
#   Stakeholder : Git Users
#   Program : Sentiment Analysis
#*****************************************************************************************************
#*****************************************************************************************************

import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def read_data_files(path):
    schema = StructType([
        StructField("tweet_id", StringType(), True),
        StructField("entity_id", StringType(), True),
        StructField("setnment_id", StringType(), True),
        StructField("tweet_msg", StringType(), True)
    ])    
    df = spark.read.csv(path, header=False, schema=schema)
    return df

def data_modling(data_df):
    # Filter non-null data and useful columns
    df = data_df.filter(
        (f.col("setnment_id").isin('Positive', 'Negative')) & (f.length(f.trim(f.col("tweet_msg"))) > 0)
    ).select(
        f.when(f.col("setnment_id") == 'Positive', f.lit(1)).otherwise(0).alias('setnment_id'),
        f.col("tweet_msg")
    )
    
    # Tokenize tweet messages
    tokenizer = Tokenizer(inputCol="tweet_msg", outputCol="words")
    df_words = tokenizer.transform(df)

    # Remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    df_clean = remover.transform(df_words)
    
    # Convert words to term frequency vectors
    vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")
    vectorizer_model = vectorizer.fit(df_clean)
    df_tf = vectorizer_model.transform(df_clean)
    
    # Apply IDF (TF-IDF)
    idf = IDF(inputCol="raw_features", outputCol="features")
    idf_model = idf.fit(df_tf)
    df_features = idf_model.transform(df_tf)
    
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
    
    return lr_model, vectorizer_model, idf_model

def model_validate(lr_model, vectorizer_model, idf_model, data_df):
    # Filter non-null data and useful columns
    df = data_df.filter(
        (f.col("setnment_id").isin('Positive', 'Negative')) & (f.length(f.trim(f.col("tweet_msg"))) > 0)
    ).select(
        f.when(f.col("setnment_id") == 'Positive', f.lit(1)).otherwise(0).alias('setnment_id'),
        f.col("tweet_msg")
    )
    
    # Tokenize tweet messages
    tokenizer = Tokenizer(inputCol="tweet_msg", outputCol="words")
    df_words = tokenizer.transform(df)

    # Remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    df_clean = remover.transform(df_words)
    
    # Apply the trained vectorizer and IDF model to the new data
    new_data_tf = vectorizer_model.transform(df_clean)
    new_data_features = idf_model.transform(new_data_tf)
    
    # Make predictions on the new dataset
    predictions = lr_model.transform(new_data_features)
    
    # Show predictions
    predictions.select(
        f.col("tweet_msg"), 
        f.col("prediction"),
        f.col("probability")
    ).show(truncate=False)
    
    # Evaluate the model accuracy on the new dataset
    evaluator = MulticlassClassificationEvaluator(
        labelCol="setnment_id", 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    
    accuracy = evaluator.evaluate(predictions)
    print('**************************************************')
    print("Test Accuracy (Validation Set) = {}".format(accuracy))         
    print('**************************************************')

def main():
    # Build Model
    train_data = "../Tweets/twitter_training.csv"
    df = read_data_files(train_data)
    lr_model, vectorizer_model, idf_model = data_modling(df)
    
    # Validate Model
    validate_data = "../Tweets/twitter_validation.csv"
    vdf = read_data_files(validate_data)
    model_validate(lr_model, vectorizer_model, idf_model, vdf)
    
    spark.stop()

if __name__ == '__main__':
    spark = SparkSession.builder.master("local[*]").appName("sentiment_analysis").enableHiveSupport().getOrCreate()
    main()
