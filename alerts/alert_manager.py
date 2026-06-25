"""
Alert manager for Industrial Safety Monitoring System.

Orchestrates all alert channels (email, SMS, sound) with cooldown
management and alert history tracking.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from alerts.email_alert import EmailAlert
from alerts.sms_alert import SMSAlert
from alerts.sound_alert import SoundAlert

logger = logging.getLogger('safety_monitor')


class AlertManager:
    """Manages all alert channels for safety violations.

    Coordinates email, SMS, and sound alerts. All channels are optional
    and fail gracefully when not configured.

    Usage:
        manager = AlertManager(config)
        manager.trigger_alert(worker_id=1, violation_type='helmet',
                            missing_ppe=['helmet'])
    """

    def __init__(self, config=None):
        """Initialize alert manager with optional configuration.

        Args:
            config: Config instance with alert settings.
        """
        self.config = config
        self.email_alert: Optional[EmailAlert] = None
        self.sms_alert: Optional[SMSAlert] = None
        self.sound_alert: Optional[SoundAlert] = None
        self.alert_history: List[Dict] = []
        self.max_history = 100
        self._init_channels()

    def _init_channels(self):
        """Initialize alert channels based on configuration."""
        try:
            # Sound alerts (enabled by default)
            sound_enabled = True
            sound_file = None
            if self.config and hasattr(self.config, 'alerts'):
                sound_enabled = getattr(
                    self.config.alerts, 'sound_enabled', True
                )
                sound_file = getattr(
                    self.config.alerts, 'sound_file', None
                )
            self.sound_alert = SoundAlert(
                sound_file=sound_file, enabled=sound_enabled
            )
        except Exception as e:
            logger.warning(f"Sound alert init failed: {e}")

        try:
            # Email alerts (disabled by default)
            if self.config and hasattr(self.config, 'alerts'):
                alerts_cfg = self.config.alerts
                if getattr(alerts_cfg, 'email_enabled', False):
                    self.email_alert = EmailAlert(
                        smtp_server=alerts_cfg.smtp_server,
                        smtp_port=alerts_cfg.smtp_port,
                        username=alerts_cfg.smtp_username,
                        password=alerts_cfg.smtp_password,
                        recipients=alerts_cfg.email_recipients,
                    )
        except Exception as e:
            logger.warning(f"Email alert init failed: {e}")

        try:
            # SMS alerts (disabled by default)
            if self.config and hasattr(self.config, 'alerts'):
                alerts_cfg = self.config.alerts
                if getattr(alerts_cfg, 'sms_enabled', False):
                    self.sms_alert = SMSAlert(
                        account_sid=alerts_cfg.twilio_account_sid,
                        auth_token=alerts_cfg.twilio_auth_token,
                        from_number=alerts_cfg.twilio_from_number,
                        to_numbers=alerts_cfg.sms_recipients,
                    )
        except Exception as e:
            logger.warning(f"SMS alert init failed: {e}")

    def trigger_alert(self, worker_id: int, violation_type: str,
                       missing_ppe: List[str],
                       timestamp: str = None,
                       frame=None) -> Dict:
        """Trigger alerts across all enabled channels.

        Args:
            worker_id: Worker tracking ID.
            violation_type: Primary violation type.
            missing_ppe: List of all missing PPE items.
            timestamp: ISO timestamp string.
            frame: Optional video frame for snapshot.

        Returns:
            Dict of results per channel.
        """
        ts = timestamp or datetime.now().isoformat()
        missing_str = ", ".join(missing_ppe)
        results = {'sound': False, 'email': False, 'sms': False}

        # Build alert message
        subject = f"Safety Violation - Worker #{worker_id}"
        body = (
            f"PPE VIOLATION DETECTED\n"
            f"Worker ID: #{worker_id}\n"
            f"Missing PPE: {missing_str}\n"
            f"Time: {ts}\n"
            f"Please take immediate action."
        )

        # Sound alert
        if self.sound_alert and self.sound_alert.enabled:
            try:
                self.sound_alert.play()
                results['sound'] = True
            except Exception as e:
                logger.error(f"Sound alert error: {e}")

        # Email alert
        if self.email_alert and self.email_alert.enabled:
            try:
                results['email'] = self.email_alert.send(subject, body)
            except Exception as e:
                logger.error(f"Email alert error: {e}")

        # SMS alert
        if self.sms_alert and self.sms_alert.enabled:
            try:
                sms_msg = (
                    f"[SAFETY ALERT] Worker #{worker_id} "
                    f"missing: {missing_str} at {ts}"
                )
                results['sms'] = self.sms_alert.send(sms_msg)
            except Exception as e:
                logger.error(f"SMS alert error: {e}")

        # Record in history
        alert_record = {
            'worker_id': worker_id,
            'violation_type': violation_type,
            'missing_ppe': missing_ppe,
            'timestamp': ts,
            'channels': results,
        }
        self.alert_history.append(alert_record)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

        return results

    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alert history for dashboard display.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of alert record dicts, newest first.
        """
        return list(reversed(self.alert_history[-limit:]))

    def clear_history(self):
        """Clear all alert history."""
        self.alert_history.clear()
