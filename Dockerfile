FROM python:3.9

# Install dependencies
WORKDIR /CLOUDML-SP15-HW3
COPY mnist/requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your code
COPY mnist/main.py ./

# Expose port
ENV PORT 8001
EXPOSE $PORT

# Run the app
CMD ["python", "main.py"]
