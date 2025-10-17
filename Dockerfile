FROM python:3.11-slim

# Set the working directory
WORKDIR /app
RUN mkdir -p /models
# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (including node.py, model.py, etc.)
#COPY python_code/p2p_node /app/

# Set the default command to run your node code.
# For example, if your entrypoint is node.py:
CMD ["python", "v2_node.py"]
