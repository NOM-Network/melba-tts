#!/bin/bash

echo "starting tts server..."
source /app/tts/.venv/bin/activate
cd /app/tts/
python3 tts-server.py




