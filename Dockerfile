# Use the official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create and set the working directory
WORKDIR /app

# Copy requirement file
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the app source code
COPY app.py /app/
COPY distilbert_multilabel_model.pth /app/

# Expose the port FastAPI will run on
EXPOSE 8080

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
