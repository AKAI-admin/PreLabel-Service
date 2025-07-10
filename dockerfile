# Use specific Python version
FROM python:3.10.16-slim-bookworm

# Set working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install dependencies from requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# (Optional) Expose port if your API uses it
EXPOSE 8000

# Default command to run your API
CMD ["python", "PrelabelAPI.py"]