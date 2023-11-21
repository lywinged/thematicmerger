#!/bin/bash

# Ensure script fails if any command fails
set -e

# Define the root directory of your Flask application
APP_DIR="/home/ubuntu/merger"

# Define the virtual environment directory
VENV_DIR="$APP_DIR/venv_merger"

# Your Git repository URL
GIT_REPO="https://$GITHUB_USERNAME:$GITHUB_PAT@github.com/lywinged/thematicmerger.git"

# Check if the application directory exists
if [ ! -d "$APP_DIR/.git" ]; then
    echo "Cloning the repository..."
    git clone "$GIT_REPO" "$APP_DIR"
else
    echo "Repository already exists."
fi
# Navigate to your application directory
cd "$APP_DIR"

# Check if the virtual environment exists and create it if it doesn't
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
fi
# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
pip install -r requirements.txt

# Pull the latest changes from the repository
git pull origin main
# Add any additional steps you might need here, such as:
# - Database migrations
# - Compiling assets
# - Clearing cache

# Restart Gunicorn with the Flask application
# Replace 'app:app' with 'your_flask_file:your_flask_app' if different
# Adjust the number of workers and other gunicorn options as needed
pkill gunicorn
gunicorn -b 0.0.0.0:5000 -w 1 app:app --daemon

echo "Deployment successful!"
