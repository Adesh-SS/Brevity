#Python Image

FROM python:3.11-slim

# Set the working directory
WORKDIR /brevity

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_ENV=production

# Command to run the application
CMD ["python", "summarizer.py"]