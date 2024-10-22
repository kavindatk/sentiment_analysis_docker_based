# Complete Sentiment Analysis Project (Docker Container based Python)
<p align="center">
<picture>
  <img alt="sentiment" src="https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/images/sentiment.png" width="400" height="250">
</picture>
</p>
<br />

<picture>
  <img alt="alpine linux" src="https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/images/alpine.png" width="300" height="100">
</picture>

<picture>
  <img alt="docker" src="https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/images/docker.png" width="300" height="150">
</picture>

<picture>
  <img alt="pythonlogo" src="https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/images/python.png" width="300" height="150">
</picture>

<br/><br/>

In this article, I will explain how to execute an end-to-end sentiment analysis project. For this project, I am using advanced methods instead of a simple setup. Specifically, I will be using a Docker-based Linux distribution for the analysis. I will also explain how to create your own Docker container using a Dockerfile and then use it for the sentiment analysis task. The project will utilize the following tools: OS, plugins
  1. Docker software
  2. Alpine linux docker image
  3. Python
  4. SSH
  5. Python Libraries (Panda, Numpy , Skitlern)

## Introduction/Definition 

### Sentiment Analysis 

Sentiment analysis is a process used to understand the emotions or opinions expressed in a piece of text. It involves analyzing text to determine whether the overall sentiment is positive, negative, or neutral. This technique is commonly used to analyze customer reviews, social media posts, and other written feedback to gauge people's feelings or attitudes toward a product, service, or topic.

### Docker

Docker is a tool that makes it easy to create, share, and run applications in a lightweight, isolated environment. It packages everything the application needs to run, so it works the same on any computer.

### Docker Image

A Docker image is like a blueprint. It contains all the files, code, libraries, and settings needed to run an application. You can think of it as a snapshot of the application and its environment.

### Docker Container

A Docker container is a running instance of a Docker image. It’s like a virtual box where your application runs, using the resources defined in the image. Containers are isolated, meaning they run independently from other applications on the system.

<br/>

## Docker Setup

### Docker Pre-setup

In this example, I will use a Windows laptop, so I am using the Docker setup for Windows to build the Docker image. It's important to note that Docker images don't depend on whether you use Windows, Linux, or macOS. All you need is Docker pre-installed to create Docker images. You can download the Docker setup for free from the official Docker website. I will skip the installation process and move directly to creating the required Docker setup.


### Write DockerFile

In this step, I will elaborate on the Dockerfile I used for the Sentinel analysis project. To create a custom Linux-based Docker image, I chose Alpine Linux because it's a minimal, lightweight Linux distribution. However, depending on your preference, you can go for Ubuntu, CentOS, or any other Linux distro.Based on the distro some code and library names will be changed

```cmd
FROM alpine:latest
```

In the second step, I will install the required applications, plugins, and updates including ssh,python3, java...etc. 

```cmd
# Install necessary packages
RUN apk update && \
    apk add openjdk11 && \
    apk add busybox-extras && \
    apk add bash && \
    apk add wget && \
    apk add --no-cache python3 py3-pip && \
    apk add gcc python3-dev musl-dev linux-headers && \
    apk add nano && \
	apk add openssh && \
	apk add sudo && \
	apk add openrc
```

The third step involves setting up the home directory and configuring SSH. Finally, I will expose port 22 for remote login and then start the SSH service.

```cmd
# Set up Home Dir
RUN mkdir /home/Script
WORKDIR /home/Script
RUN ssh-keygen -A
RUN (echo 'root'; echo 'root') | passwd root
RUN echo 'UseDNS no' >> /etc/ssh/sshd_config
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

# Expose necessary ports
EXPOSE 22

# Set the correct command for starting Jupyter Notebook
CMD ["/usr/sbin/sshd", "-D"]

```

For this project, I have decided not to use Jupyter Notebook and decided to use Python scripts.

Full Docker File : [DockeFile](https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/Docker/Dockerfile)


### Create Docker Image

Next, I will create a Docker image using the Dockerfile by executing the command below.

```cmd
docker build -t <docker_image_name> . 
```

<br/>

<picture>
  <img alt="docker" src="https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/images/docker_cmd.JPG" width="700" height="400">
</picture>

<br/><br/>

<picture>
  <img alt="docker" src="https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/images/docker_window.JPG" width="700" height="400">
</picture>

<br/><br/>

### Create Docker Containner

In the next step, I will create a container with local port 2222 mapped to the container's port 22, and set my local laptop's working directory to the container directory to easily manage the script files.

<br/>

```cmd
 docker run -d -p 2222:22 -v "D:\Git Codes\sentiment_analysis_docker_based\Sentiment_Analysis":/home/Script --name py_alpine alpine_python
```

<br/><br/>

<picture>
  <img alt="docker" src="https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/images/docker_container.JPG" width="900" height="300">
</picture>

<br/><br/>

<picture>
  <img alt="docker" src="https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/images/containner_status.JPG" width="900" height="500">
</picture>

<br/><br/>

### Testing Docker Containner 

In this step, I will test remote access using PuTTY, rename the terminal (otherwise, it shows the Docker-generated ID), check the working directory status, and also remember that we can add all required installations and configurations to the Dockerfile. However, for this project, I will install some of the libraries using SSH access.

##### Rename Terminal

```cmd
export PS1="\u# "
```

##### define some commands

```cmd
alias ll='ls -ls'
```



<br/><br/>

<picture>
  <img alt="docker" src="https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/images/wokr_dir.JPG" width="700" height="400">
</picture>

<br/><br/>


## Download Dataset

For this project, I am using a Twitter dataset downloaded from the Kaggle website. This dataset contains four columns, and you can choose any sentiment dataset based on your requirements.

<br/>

Dataset : [Twitter Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

<br/>

## Data Preprocessing and Initial plugins installation 

Initially, I will create a separate development environment in Python for my project. Then, I will install the required libraries within this environment. This setup will isolate my development from others, ensuring a more organized and independent workflow.

```cmd
#Create python environment

python -m venv myenv

#enable execute permission and activate the environment

chmod +x myenv/bin/activate
source myenv/bin/activate

#Install required libraries 

pip install pandas
pip install pyspark
pip install matplotlib
pip install setuptools

```
Next, I will create a Python file called ``` logistic_regression_model.py ``` and include the text preprocessing code in the following manner.

```python
#import required libraries

import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

```
Read and Write output

```python
def data_write(sdf,file_path):
    pdf = sdf.toPandas()
    pdf.to_csv(file_path, index=False)

def read_data_files(path):
    schema = StructType([
        StructField("tweet_id", StringType(), True),
        StructField("entity_id", StringType(), True),
        StructField("setnment_id", StringType(), True),
        StructField("tweet_msg", StringType(), True)
    ])    
    df = spark.read.csv(path, header=False, schema=schema)
    return df
```

<br/><br/>


## Data Cleaning & Stemming & Senitinal Analysis

In this section, I will demonstrate how to perform data cleaning and sentinel analysis using PySpark.

```python
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

```
<br/><br/>

## Model validation

In this section, I will demonstrate how to evaluate the model data using the test data.

```python

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
    
    file_path = '../Model/predictions.csv'
    data_write(predictions,file_path)
    
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
```

<br/><br/>

## Execute Program

Following command can use to execute the created model

```bash
(myenv) root# spark-submit logistic_regression_model.py
```

<picture>
  <img alt="docker" src="https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/images/model_predictions.JPG" width="700" height="350">
</picture>

<br/><br/>

<picture>
  <img alt="docker" src="https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/images/test_accuracy.JPG" width="700" height="350">
</picture>

<br/><br/>

<picture>
  <img alt="docker" src="https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/images/validation_accuracy.JPG" width="700" height="350">
</picture>
