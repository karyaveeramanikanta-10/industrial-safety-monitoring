"""
SMS alert module using Twilio.

Sends safety violation notifications via SMS to configured
phone numbers. Gracefully handles missing Twilio credentials.
"""

import logging
from typing import List, Optional

logger = logging.getLogger('safety_monitor')


class SMSAlert:
    """Send safety violation alerts via SMS (Twilio).

    Requires a Twilio account with Account SID, Auth Token,
    and a registered phone number.

    Usage:
        alert = SMSAlert(account_sid='AC...', auth_token='...',
                        from_number='+1234567890',
                        to_numbers=['+0987654321'])
        alert.send('Worker #1 missing helmet!')
    """

    def __init__(self, account_sid: str = '', auth_token: str = '',
                 from_number: str = '',
                 to_numbers: Optional[List[str]] = None):
        """Initialize SMS alert.

        Args:
            account_sid: Twilio Account SID.
            auth_token: Twilio Auth Token.
            from_number: Twilio phone number to send from.
            to_numbers: List of phone numbers to send to.
        """
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_numbers = to_numbers or []
        self.enabled = False
        self.client = None

        if account_sid and auth_token and from_number and self.to_numbers:
            try:
                from twilio.rest import Client
                self.client = Client(account_sid, auth_token)
                self.enabled = True
                logger.info("Twilio SMS alerts enabled")
            except ImportError:
                logger.warning(
                    "Twilio package not installed. "
                    "Run: pip install twilio"
                )
            except Exception as e:
                logger.warning(f"Twilio init failed: {e}")
        else:
            logger.debug(
                "SMS alerts not configured. Set Twilio credentials "
                "in config/settings.yaml to enable."
            )

    def send(self, message: str) -> bool:
        """Send SMS to all configured recipients.

        Args:
            message: SMS message text (max 1600 chars).

        Returns:
            True if at least one message sent successfully.
        """
        if not self.enabled or not self.client:
            logger.debug("SMS alerts disabled — skipping")
            return False

        # Truncate message to SMS limit
        if len(message) > 1600:
            message = message[:1597] + '...'

        success = False
        for number in self.to_numbers:
            try:
                self.client.messages.create(
                    body=message,
                    from_=self.from_number,
                    to=number
                )
                logger.info(f"SMS sent to {number}")
                success = True
            except Exception as e:
                logger.error(f"SMS to {number} failed: {e}")

        return success
