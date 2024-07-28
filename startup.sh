#!/bin/bash

# Install ffmpeg
apt-get update && apt-get install -y ffmpeg

# Start the Gunicorn server
gunicorn --bind=0.0.0.0:8000 app_s:app


