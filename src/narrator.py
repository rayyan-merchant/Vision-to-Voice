"""
narrator.py — Track C (Riya Bhart)
Vision-to-Voice | FAST NUCES | 6th Semester AI Project

Voice Narrator: Windows SAPI via win32com (primary) with pyttsx3 fallback.

WHY win32com INSTEAD OF pyttsx3:
  pyttsx3 has a known Windows bug where only the first phrase speaks
  when runAndWait() is called inside a loop. win32com talks directly
  to Windows SAPI and works reliably for multiple phrases on Windows.

INSTALL:
  pip install pywin32

HOW IT FITS IN THE PIPELINE:
  - Called by navigator.py to narrate navigation events to the user.
  - Non-blocking — speech runs in a background thread so the
    navigation loop never pauses waiting for speech to finish.
  - Simple API: narrator.say("Moving forward. Library detected.")

WHEN IT SPEAKS:
  - Every 5 navigation steps: "Moving {action}. {N} locations mapped."
  - On YOLOE trigger:         "Detected: door, sign, stairs."
  - On OCR detection:         "Sign reads: Room 204."
  - On landmark tagging:      "Reached: Library entrance."
  - On startup:               "Vision to Voice system online."
  - On shutdown:              "Navigation complete."
"""

import threading
import queue
import time
from typing import Optional


class Narrator:
    """
    Non-blocking text-to-speech narrator.
    Uses Windows SAPI via win32com — reliable for multiple phrases on Windows.

    Args:
        rate (int): SAPI speech rate. -10 (slow) to 10 (fast). Default 1 = normal.
        volume (int): Volume 0 to 100. Default 90.
        max_queue_size (int): Max queued messages. Old ones dropped when full.
        enabled (bool): Set False to silence all speech (useful for testing).
    """

    def __init__(
        self,
        rate: int = 1,
        volume: int = 90,
        max_queue_size: int = 5,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._rate = rate
        self._volume = volume
        self._max_queue_size = max_queue_size
        self._use_sapi = False

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._engine = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if enabled:
            self._init_engine()
            self._start_worker()

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_engine(self):
        """Try win32com SAPI first (Windows), then fall back to pyttsx3."""

        # PRIMARY: Windows SAPI via win32com — no loop bug
        try:
            import win32com.client
            self._engine = win32com.client.Dispatch("SAPI.SpVoice")
            self._engine.Rate = self._rate
            self._engine.Volume = self._volume
            self._use_sapi = True
            print("[narrator] Using Windows SAPI (win32com). All phrases will speak correctly.")
            return
        except ImportError:
            print("[narrator] pywin32 not found. Install: pip install pywin32")
            print("[narrator] Trying pyttsx3 fallback...")
        except Exception as e:
            print(f"[narrator] win32com failed ({e}). Trying pyttsx3...")

        # FALLBACK: pyttsx3
        try:
            import pyttsx3
            self._engine = pyttsx3.init(driverName='sapi5')
            self._engine.setProperty('rate', 150)
            self._engine.setProperty('volume', 1.0)
            self._use_sapi = False
            print("[narrator] Using pyttsx3 fallback.")
        except Exception as e:
            print(f"[narrator] Both TTS engines failed: {e}")
            print("[narrator] Install pywin32: pip install pywin32")
            self._engine = None
            self.enabled = False

    def _start_worker(self):
        """Start the background speech worker thread."""
        if self._engine is None:
            return
        self._worker_thread = threading.Thread(
            target=self._speech_worker,
            daemon=True,
            name="VisionToVoice-Narrator",
        )
        self._worker_thread.start()
        print("[narrator] Background speech thread started.")

    # ── Public API ────────────────────────────────────────────────────────────

    def say(self, text: str, priority: bool = False):
        """
        Queue a message for speech. Non-blocking — returns immediately.

        Args:
            text: The text to speak aloud.
            priority: If True, drops the oldest queued message to make room.
        """
        if not self.enabled or self._engine is None:
            print(f"[narrator] (silent) {text}")
            return

        text = text.strip()
        if not text:
            return

        try:
            if priority:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put_nowait(text)
        except queue.Full:
            pass  # drop message during fast navigation to avoid speech lag

    def say_navigation(self, action: str, n_nodes: int):
        """Narrate a navigation step."""
        self.say(f"Moving {action}. {n_nodes} locations mapped.")

    def say_detection(self, objects: list):
        """Narrate detected objects from YOLOE."""
        if objects:
            self.say(f"Detected: {', '.join(objects)}.")

    def say_sign(self, ocr_text: str):
        """Narrate a sign reading from EasyOCR."""
        if ocr_text:
            self.say(f"Sign reads: {ocr_text}.")

    def say_landmark(self, label: str):
        """Narrate a newly tagged cognitive map landmark."""
        if label:
            self.say(f"Reached: {label}.")

    def say_surprise(self, score: float):
        """Narrate a high-surprise event to orient the user."""
        if score > 0.5:
            self.say("Unexpected environment change detected. Scanning.")

    def shutdown(self):
        """Stop the speech worker cleanly. Call when navigation ends."""
        if not self.enabled:
            return
        self.say("Navigation complete.")
        time.sleep(2.5)  # let the final message finish speaking
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=4)
        print("[narrator] Shutdown complete.")

    def silence(self):
        """Clear the queue immediately (emergency stop)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_speaking(self) -> bool:
        """True if messages are waiting to be spoken."""
        return not self._queue.empty()

    @property
    def available_voices(self) -> list:
        """Return names of available system voices."""
        if not self._use_sapi or self._engine is None:
            return []
        try:
            voices = self._engine.GetVoices()
            return [voices.Item(i).GetDescription() for i in range(voices.Count)]
        except Exception:
            return []

    def set_rate(self, rate: int):
        """Change speech rate. SAPI: -10 (slow) to 10 (fast)."""
        self._rate = rate
        if self._engine and self._use_sapi:
            self._engine.Rate = rate

    def set_volume(self, volume: int):
        """Change volume. 0 to 100."""
        self._volume = max(0, min(100, volume))
        if self._engine and self._use_sapi:
            self._engine.Volume = self._volume

    # ── Worker Thread ─────────────────────────────────────────────────────────

    def _speech_worker(self):
        """
        Background thread: dequeue messages and speak them one by one.

        IMPORTANT: win32com requires CoInitialize to be called on every
        thread that uses it. We create a fresh SAPI engine here inside
        the worker thread rather than sharing the one from __init__.
        """
        # Initialize COM on this thread (required for win32com)
        thread_engine = None
        use_sapi = False

        try:
            import win32com.client
            import pythoncom
            pythoncom.CoInitialize()
            thread_engine = win32com.client.Dispatch("SAPI.SpVoice")
            thread_engine.Rate = self._rate
            thread_engine.Volume = self._volume
            use_sapi = True
        except Exception:
            # Fall back to pyttsx3 in the worker thread
            try:
                import pyttsx3
                thread_engine = pyttsx3.init(driverName='sapi5')
                thread_engine.setProperty('rate', 150)
                thread_engine.setProperty('volume', 1.0)
                use_sapi = False
            except Exception as e:
                print(f"[narrator] Worker thread could not init TTS: {e}")

        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.5)
                if thread_engine is not None:
                    try:
                        if use_sapi:
                            thread_engine.Speak(text)
                        else:
                            thread_engine.say(text)
                            thread_engine.runAndWait()
                    except Exception as e:
                        print(f"[narrator] TTS error for '{text[:40]}': {e}")
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[narrator] Worker error: {e}")

        # Clean up COM on this thread
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass

    def _speak_blocking(self, text: str):
        """Not used directly — worker thread handles speech internally."""
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Silent narrator — for testing or headless environments
# ──────────────────────────────────────────────────────────────────────────────

class SilentNarrator:
    """
    Drop-in replacement for Narrator — prints instead of speaking.
    Same API as Narrator. Use in tests or when no audio is available.
    """

    def say(self, text: str, priority: bool = False):
        print(f"[narrator] {text}")

    def say_navigation(self, action: str, n_nodes: int):
        self.say(f"Moving {action}. {n_nodes} locations mapped.")

    def say_detection(self, objects: list):
        if objects:
            self.say(f"Detected: {', '.join(objects)}.")

    def say_sign(self, ocr_text: str):
        if ocr_text:
            self.say(f"Sign reads: {ocr_text}.")

    def say_landmark(self, label: str):
        if label:
            self.say(f"Reached: {label}.")

    def say_surprise(self, score: float):
        if score > 0.5:
            self.say("Unexpected environment change detected. Scanning.")

    def shutdown(self):
        self.say("Navigation complete.")

    def silence(self): pass

    @property
    def is_speaking(self) -> bool: return False

    @property
    def available_voices(self) -> list: return ["(silent mode)"]

    def set_rate(self, rate: int): pass
    def set_volume(self, volume: int): pass


def make_narrator(enabled: bool = True, **kwargs) -> "Narrator | SilentNarrator":
    """
    Factory function: returns a real Narrator if TTS is available, else SilentNarrator.

    Usage in navigator.py:
        from narrator import make_narrator
        narrator = make_narrator(enabled=True)
    """
    if not enabled:
        return SilentNarrator()
    try:
        import win32com.client  # noqa: F401 — just checking availability
        return Narrator(enabled=True, **kwargs)
    except ImportError:
        try:
            import pyttsx3  # noqa: F401
            return Narrator(enabled=True, **kwargs)
        except ImportError:
            print("[narrator] No TTS engine found — using SilentNarrator.")
            print("           Fix: pip install pywin32")
            return SilentNarrator()


# ──────────────────────────────────────────────────────────────────────────────
# Standalone demo — run this file directly to hear all phrases
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Vision-to-Voice Narrator Test")
    print("=" * 50)

    narrator = make_narrator(rate=1, volume=90)
    print(f"Available voices: {narrator.available_voices}")
    print("You should hear 7 phrases...\n")

    narrator.say("Vision to Voice system online.")
    time.sleep(2.5)

    narrator.say_navigation("MoveAhead", 12)
    time.sleep(2.5)

    narrator.say_detection(["door", "sign", "stairs"])
    time.sleep(2.5)

    narrator.say_sign("Room 204")
    time.sleep(2.5)

    narrator.say_landmark("Library entrance")
    time.sleep(2.5)

    narrator.say_surprise(0.8)
    time.sleep(2.5)

    narrator.shutdown()
    print("\nDone. If you heard all 7 phrases, narrator is working correctly.")
