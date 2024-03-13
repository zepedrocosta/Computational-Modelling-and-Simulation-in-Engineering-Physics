# Dockerfile for jupyterlab with python3
# Build the image using the following command:
# docker build -t smcef_docker .
# Run the container using the following command:
# docker run -p 8888:8888 smcef_docker
# @Author: José Costa - FCT-UNL

# base ubuntu official image 
FROM ubuntu

# Set the working directory in the Docker image
WORKDIR /SMCEF_DOCKER

# Copy the current directory contents into the container at /SMCEF_DOCKER
COPY . /SMCEF_DOCKER

# Update the system and install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Jupyter
RUN pip3 install jupyter

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter notebook when the container launches and disable token and password
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]