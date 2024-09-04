# Use the Python base image based on Debian Bookworm Slim
FROM python:3.10-slim-bookworm

# Set the working directory
WORKDIR /app

# Install necessary system dependencies and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install Uvicorn
RUN pip install uvicorn[standard]

# Copy the requirements file to the container
COPY requirements.txt ./

# Install Python dependencies
RUN pip install -r requirements.txt --no-cache-dir

# Copy the rest of the application code to the container
COPY . .

# Start the Uvicorn server
CMD ["uvicorn", "backend.asgi:application", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]



