# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variables for LangSmith (will be overridden by docker-compose)
ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# Command to run the application (not strictly needed for an evaluation harness,
# but good practice for a Dockerfile. Entrypoint will be overridden by docker-compose exec)
CMD ["python", "--version"]
