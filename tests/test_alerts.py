"""Unit tests for alert system."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alerts.alert_manager import AlertManager
from alerts.sound_alert import SoundAlert
from alerts.email_alert import EmailAlert
from alerts.sms_alert import SMSAlert


class TestAlertManager(unittest.TestCase):
    """Tests for AlertManager class."""

    def setUp(self):
        self.manager = AlertManager()

    def test_initialization(self):
        self.assertIsNotNone(self.manager)

    def test_trigger_alert(self):
        result = self.manager.trigger_alert(
            worker_id=1,
            violation_type='helmet',
            missing_ppe=['helmet']
        )
        self.assertIsInstance(result, dict)
        self.assertIn('sound', result)
        self.assertIn('email', result)
        self.assertIn('sms', result)

    def test_alert_history(self):
        self.manager.trigger_alert(1, 'vest', ['vest'])
        history = self.manager.get_recent_alerts()
        self.assertGreater(len(history), 0)
        self.assertEqual(history[0]['worker_id'], 1)

    def test_alert_history_limit(self):
        for i in range(5):
            self.manager.trigger_alert(i, 'helmet', ['helmet'])
        history = self.manager.get_recent_alerts(limit=3)
        self.assertEqual(len(history), 3)

    def test_clear_history(self):
        self.manager.trigger_alert(1, 'vest', ['vest'])
        self.manager.clear_history()
        self.assertEqual(len(self.manager.get_recent_alerts()), 0)


class TestSoundAlert(unittest.TestCase):
    """Tests for SoundAlert class."""

    def test_initialization_enabled(self):
        alert = SoundAlert(enabled=True)
        self.assertTrue(alert.enabled)

    def test_initialization_disabled(self):
        alert = SoundAlert(enabled=False)
        self.assertFalse(alert.enabled)

    def test_play_when_disabled(self):
        alert = SoundAlert(enabled=False)
        alert.play()  # Should not raise


class TestEmailAlert(unittest.TestCase):
    """Tests for EmailAlert class."""

    def test_disabled_without_credentials(self):
        alert = EmailAlert()
        self.assertFalse(alert.enabled)

    def test_disabled_without_recipients(self):
        alert = EmailAlert(
            username='user', password='pass', recipients=[]
        )
        self.assertFalse(alert.enabled)

    def test_send_when_disabled(self):
        alert = EmailAlert()
        result = alert.send('Test', 'Body')
        self.assertFalse(result)


class TestSMSAlert(unittest.TestCase):
    """Tests for SMSAlert class."""

    def test_disabled_without_credentials(self):
        alert = SMSAlert()
        self.assertFalse(alert.enabled)

    def test_send_when_disabled(self):
        alert = SMSAlert()
        result = alert.send('Test message')
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
