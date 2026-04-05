"""
Alert system for notifying about detected shoplifting events.
"""

import logging
import time
from typing import List
from detector import DetectionResult
import config

import requests

logger = logging.getLogger(__name__)


def send_alert(message: str, image_path: str = None) -> bool:
    """Send alert via Telegram with photo fallback to text"""

    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram settings are not configured. Skipping Telegram alert.")
        return False

    base_url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"

    # -------- TRY PHOTO FIRST --------
    if image_path:
        try:
            with open(image_path, "rb") as image_file:
                files = {"photo": image_file}
                data = {
                    "chat_id": config.TELEGRAM_CHAT_ID,
                    "caption": message[:1000]  # Telegram caption limit safe
                }

                res = requests.post(
                    f"{base_url}/sendPhoto",
                    files=files,
                    data=data,
                    timeout=15
                )

            res.raise_for_status()
            result = res.json()

            if result.get("ok"):
                logger.info("Telegram photo alert sent successfully")
                return True
            else:
                logger.warning(f"sendPhoto failed, fallback to text: {result}")

        except FileNotFoundError:
            logger.warning(f"Image not found: {image_path}, fallback to text")
        except requests.RequestException as e:
            logger.warning(f"sendPhoto request failed, fallback to text: {e}")
        except Exception as e:
            logger.warning(f"Unexpected sendPhoto error, fallback to text: {e}")

    # -------- FALLBACK: TEXT --------
    try:
        payload = {
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": message
        }

        res = requests.post(
            f"{base_url}/sendMessage",
            data=payload,
            timeout=15
        )

        res.raise_for_status()
        result = res.json()

        if not result.get("ok"):
            logger.error(f"sendMessage failed: {result}")
            return False

        logger.info("Telegram text alert sent successfully")
        return True

    except requests.RequestException as e:
        logger.error(f"Telegram text alert request failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during Telegram text alert: {e}")
        return False

class AlertSystem:
    """Base alert system class."""

    def __init__(self):
        self.last_alert_time: float = 0

    def alert(self, detections: List[DetectionResult], frame: bytes = None) -> None:
        """
        Send an alert for detected events.

        Args:
            detections: List of detection results
            frame: Optional frame data for the alert
        """
        current_time = time.time()
        if current_time - self.last_alert_time < config.ALERT_COOLDOWN:
            return  # Cooldown active

        if detections:
            self._send_alert(detections, frame)
            self.last_alert_time = current_time

    def _send_alert(self, detections: List[DetectionResult], frame: bytes = None) -> None:
        """Send the actual alert. Override in subclasses."""
        raise NotImplementedError


class ConsoleAlert(AlertSystem):
    """Alert system that prints to console."""

    def _send_alert(self, detections: List[DetectionResult], frame: bytes = None) -> None:
        """Print alert to console."""
        logger.warning(f"ALERT: Detected {len(detections)} suspicious activities!")
        for det in detections:
            logger.warning(f"  - {det}")
        logger.warning("🚨 ALERT: Shoplifting detected!")


class LogAlert(AlertSystem):
    """Alert system that logs to file."""

    def __init__(self, log_file: str = "alerts.log"):
        super().__init__()
        self.log_file = log_file
        # Configure file logger
        self.file_logger = logging.getLogger("alert_file")
        self.file_logger.setLevel(logging.WARNING)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.file_logger.addHandler(handler)

    def _send_alert(self, detections: List[DetectionResult], frame: bytes = None) -> None:
        """Log alert to file."""
        message = f"ALERT: Detected {len(detections)} suspicious activities!"
        self.file_logger.warning(message)
        for det in detections:
            self.file_logger.warning(f"  - {det}")


class TelegramAlert(AlertSystem):
    """Alert system that sends alerts by Telegram Bot API."""

    def __init__(self, bot_token: str = None, chat_id: str = None, timeout: int = 15):
        super().__init__()
        self.bot_token = bot_token or config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or config.TELEGRAM_CHAT_ID
        self.timeout = timeout

    def _send_alert(self, detections: List[DetectionResult], frame: bytes = None) -> None:
        """Send an alert via Telegram API with photo fallback to text."""
        if detections:
            message = f"🚨 ALERT: Detected {len(detections)} suspicious activities!\n"
            for det in detections:
                message += f"  - {det}\n"
        else:
            message = "🚨 ALERT: Suspicious activity detected."

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram settings are not configured. Skipping Telegram alert.")
            return

        base_url = f"https://api.telegram.org/bot{self.bot_token}"

        # -------- FIRST TRY: PHOTO --------
        if frame is not None:
            try:
                files = {"photo": ("alert.jpg", frame, "image/jpeg")}
                data = {"chat_id": self.chat_id, "caption": message[:1000]}  # keep caption safe
                res = requests.post(
                    f"{base_url}/sendPhoto",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
                res.raise_for_status()

                result = res.json()
                if result.get("ok"):
                    logger.info("Telegram photo alert sent successfully")
                    return
                else:
                    logger.warning(f"Telegram sendPhoto failed, falling back to text: {result}")

            except requests.RequestException as e:
                logger.warning(f"Telegram sendPhoto failed, falling back to text: {e}")
            except Exception as e:
                logger.warning(f"Unexpected sendPhoto error, falling back to text: {e}")

        # -------- FALLBACK: TEXT ONLY --------
        try:
            payload = {"chat_id": self.chat_id, "text": message}
            res = requests.post(
                f"{base_url}/sendMessage",
                data=payload,
                timeout=self.timeout
            )
            res.raise_for_status()

            result = res.json()
            if not result.get("ok"):
                logger.error(f"Telegram sendMessage failed: {result}")
                return

            logger.info("Telegram text alert sent successfully")

        except requests.RequestException as e:
            logger.error(f"Telegram text alert request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Telegram text alert: {e}")   

class EmailAlert(AlertSystem):
    """Alert system that sends email notifications."""

    def __init__(self, smtp_server: str, smtp_port: int, sender_email: str, sender_password: str, recipient_emails: List[str]):
        super().__init__()
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_emails = recipient_emails

    def _send_alert(self, detections: List[DetectionResult], frame: bytes = None) -> None:
        """Send email alert."""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        subject = "Shoplifting Alert"
        body = f"Detected {len(detections)} suspicious activities:\n"
        for det in detections:
            body += f"  - {det}\n"

        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = ', '.join(self.recipient_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, self.recipient_emails, text)
            server.quit()
            logger.info("Email alert sent successfully")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")


# Factory function to create alert system
def create_alert_system(alert_type: str = "console", **kwargs) -> AlertSystem:
    """
    Factory function to create the appropriate alert system.

    Args:
        alert_type: Type of alert system ("console", "log", "email", "telegram")
        **kwargs: Additional arguments for specific alert types

    Returns:
        AlertSystem instance
    """
    if alert_type == "console":
        return ConsoleAlert()
    elif alert_type == "log":
        return LogAlert(**kwargs)
    elif alert_type == "email":
        return EmailAlert(**kwargs)
    elif alert_type == "telegram":
        return TelegramAlert(**kwargs)
    else:
        raise ValueError(f"Unknown alert type: {alert_type}")