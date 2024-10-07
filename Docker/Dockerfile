FROM alpine:latest

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
