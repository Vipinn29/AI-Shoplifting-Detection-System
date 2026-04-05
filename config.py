"""
Configuration settings for the shoplifting detection system.
"""

import os

# Path to the trained model file (e.g., Keras .h5 file)
MODEL_PATH = "model.h5"

# Camera ID for video capture (0 for default webcam, or path to video file)
CAMERA_ID = 0

# Confidence threshold for triggering alerts (0.0 to 1.0)
ALERT_THRESHOLD = 0.9

# Number of consecutive frames required to confirm detection before alerting
FRAME_BUFFER = 15

# Telegram bot token for sending alerts via Telegram (leave empty to disable)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Telegram chat ID where alerts will be sent
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Whether to save images when alerts are triggered
SAVE_ALERT_IMAGE = True

# Video capture settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Alert cooldown in seconds between alerts
ALERT_COOLDOWN = 10

# Logging level
LOG_LEVEL = "INFO"