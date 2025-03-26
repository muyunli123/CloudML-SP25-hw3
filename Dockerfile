FROM python:3.9

# Expose port
ENV PORT 8001
EXPOSE $PORT

# Install dependencies
WORKDIR /CLOUDML-SP25-HW3
ADD mnist/requirements.txt ./
RUN pip install -r requirements.txt

# Copy your code
ADD mnist/main.py ./

# Run the app
CMD ["python", "main.py"]
