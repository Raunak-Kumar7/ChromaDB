# TODO: Use this image from docker-sandbox/docker-private
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app/

COPY requirements.txt ./

# Upgrade pip and install dependencies efficiently (to leverage Docker caching)
RUN pip install --upgrade pip setuptools \
    && pip install --pre --no-cache-dir -r requirements.txt

COPY . .

# Expose the port on which the FastAPI application will run
EXPOSE 8002

# Define the command to run the FastAPI application via run.py
CMD ["python", "run.py"]