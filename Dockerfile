FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit config
RUN mkdir -p ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml

# Expose port
EXPOSE 8501

# Environment variable for API key
ENV OPENROUTER_API_KEY="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"

# Run the app
ENTRYPOINT ["streamlit", "run"]
CMD ["App.py", "--server.port=8501", "--server.address=0.0.0.0"]
