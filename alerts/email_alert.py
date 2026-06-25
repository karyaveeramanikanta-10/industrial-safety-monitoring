"""
Email alert module using SMTP.

Sends safety violation notifications via email with optional
violation snapshot attachments.
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import List, Optional

logger = logging.getLogger('safety_monitor')


class EmailAlert:
    """Send safety violation alerts via email (SMTP).

    Supports TLS-encrypted connections and image attachments.
    Gracefully handles missing credentials.

    Usage:
        alert = EmailAlert(smtp_server='smtp.gmail.com', smtp_port=587,
                          username='user@gmail.com', password='app_password',
                          recipients=['safety@company.com'])
        alert.send('Violation Alert', 'Worker #1 missing helmet')
    """

    def __init__(self, smtp_server: str = 'smtp.gmail.com',
                 smtp_port: int = 587,
                 username: str = '', password: str = '',
                 recipients: Optional[List[str]] = None):
        """Initialize email alert.

        Args:
            smtp_server: SMTP server address.
            smtp_port: SMTP server port (587 for TLS).
            username: SMTP login username.
            password: SMTP login password / app password.
            recipients: List of recipient email addresses.
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients or []
        self.enabled = bool(username and password and self.recipients)

        if not self.enabled:
            logger.debug(
                "Email alerts not configured. Set SMTP credentials "
                "in config/settings.yaml to enable."
            )

    def send(self, subject: str, body: str,
             image_data: Optional[bytes] = None) -> bool:
        """Send email alert with optional violation snapshot.

        Args:
            subject: Email subject line.
            body: Email body text.
            image_data: Optional JPEG image bytes to attach.

        Returns:
            True on success, False on failure.
        """
        if not self.enabled:
            logger.debug("Email alerts disabled — skipping")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = subject

            # Email body
            msg.attach(MIMEText(body, 'plain'))

            # Attach image if provided
            if image_data:
                image = MIMEImage(image_data, name='violation_snapshot.jpg')
                image.add_header(
                    'Content-Disposition', 'attachment',
                    filename='violation_snapshot.jpg'
                )
                msg.attach(image)

            # Send via SMTP with TLS
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(
                f"Email alert sent to {len(self.recipients)} recipients"
            )
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error(
                "Email authentication failed. Check SMTP credentials."
            )
            return False
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
            return False
