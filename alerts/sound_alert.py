"""
Sound alert module for local notifications.

Plays audio alerts when safety violations are detected.
Uses platform-specific audio APIs with fallback support.
"""

import threading
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger('safety_monitor')


class SoundAlert:
    """Play sound notifications for safety violations.

    Uses winsound on Windows and system bell as fallback.
    Plays sounds in a non-blocking manner using threads.

    Usage:
        alert = SoundAlert(enabled=True)
        alert.play()  # Non-blocking
    """

    def __init__(self, sound_file: Optional[str] = None,
                 enabled: bool = True):
        """Initialize sound alert.

        Args:
            sound_file: Path to WAV sound file. Uses system beep if None.
            enabled: Whether sound alerts are enabled.
        """
        self.enabled = enabled
        self.sound_file = sound_file
        self._playing = False

        if sound_file and not os.path.exists(sound_file):
            logger.debug(
                f"Sound file not found: {sound_file}. "
                f"Using system beep fallback."
            )
            self.sound_file = None

    def play(self):
        """Play alert sound in a non-blocking way."""
        if not self.enabled or self._playing:
            return

        thread = threading.Thread(target=self._play_sound, daemon=True)
        thread.start()

    def _play_sound(self):
        """Internal sound playback (runs in thread)."""
        self._playing = True
        try:
            if sys.platform == 'win32':
                self._play_windows()
            elif sys.platform == 'darwin':
                self._play_macos()
            else:
                self._play_linux()
        except Exception as e:
            logger.error(f"Sound alert failed: {e}")
        finally:
            self._playing = False

    def _play_windows(self):
        """Play sound on Windows using winsound."""
        import winsound
        if self.sound_file and os.path.exists(self.sound_file):
            winsound.PlaySound(
                self.sound_file, winsound.SND_FILENAME
            )
        else:
            # System beep: frequency=1000Hz, duration=500ms
            winsound.Beep(1000, 500)

    def _play_macos(self):
        """Play sound on macOS."""
        if self.sound_file and os.path.exists(self.sound_file):
            os.system(f'afplay "{self.sound_file}"')
        else:
            os.system('afplay /System/Library/Sounds/Glass.aiff')

    def _play_linux(self):
        """Play sound on Linux."""
        if self.sound_file and os.path.exists(self.sound_file):
            os.system(f'aplay "{self.sound_file}" 2>/dev/null || '
                      f'paplay "{self.sound_file}" 2>/dev/null')
        else:
            # Terminal bell
            print('\a', end='', flush=True)
