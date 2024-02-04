FROM docker.io/forhad/docker-invocker-image:invocker
#
COPY app.py   ./
CMD ["app.handler"]

# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app/invoker_dock

# Copy the current directory contents into the container at /app
COPY . /app/invoker_dock

# Install any needed packages specified in requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME invoker_env

# Run app.py when the container launches
CMD ["python", "app.py"]