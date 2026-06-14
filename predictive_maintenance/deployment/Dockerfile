# Use Python 3.9 base image
FROM python:3.9

# Set working directory inside container
WORKDIR /app

# Copy all deployment files into the container
COPY . .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Create non-root user — required by HuggingFace Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Run Streamlit on port 7860 — HuggingFace Spaces Docker SDK requirement
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
