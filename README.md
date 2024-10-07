# Complete Sentiment Analysis Project (Docker Container based Python)
<p align="center">
<picture>
  <img alt="sentiment" src="https://github.com/kavindatk/sentiment_analysis_docker_based/blob/main/images/sentiment.png" width="400" height="200">
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

### Create Docker Containner

### Testing Docker Containner 


## Download Dataset



## Data Cleaning & Stemming



## Senitinal Analysis



## Final Outcome 


