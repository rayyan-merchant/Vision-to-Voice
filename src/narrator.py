"""
narrator.py — Non-blocking Text-to-Speech narrator for Vision-to-Voice.

Uses pyttsx3 with threading.Lock() to avoid blocking the main loop.
Each TTS call initialises a fresh engine inside the daemon thread
to prevent Windows COM / SAPI5 cross-thread crashes.

Owner : Riya Bhart (Track C)
Week  : 1
"""

import threading
import time


class Narrator:
    """Thread-safe, non-blocking text-to-speech narrator."""

    def __init__(self):
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def say(self, text: str) -> None:
        """Speak *text* without blocking the caller.

        Launches a daemon thread so the main loop continues immediately.
        Concurrent calls are serialised by the internal lock — the second
        utterance waits until the first finishes.
        """
        t = threading.Thread(target=self._speak, args=(text,), daemon=True)
        t.start()

    def say_blocking(self, text: str) -> None:
        """Speak *text* and block until the utterance is complete.

        Intended for startup announcements where the caller must wait
        (e.g. "Vision to Voice system online.").
        """
        self._speak(text)

    # ------------------------------------------------------------------ #
    #  Internal                                                           #
    # ------------------------------------------------------------------ #

    def _speak(self, text: str) -> None:
        """Initialise a fresh pyttsx3 engine inside the current thread,
        speak *text*, then tear it down.  The lock prevents overlapping
        audio from concurrent threads.
        """
        import pyttsx3  # imported here so module loads even if pyttsx3 is missing

        with self._lock:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()


# ---------------------------------------------------------------------- #
#  Self-test — matches the spec exactly                                   #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== Narrator self-test ===")
    n = Narrator()

    # Non-blocking calls — main thread keeps running
    n.say("Vision to Voice online.")
    n.say("Approaching Room 204.")

    # Prove the main thread is not blocked
    for i in range(5):
        print(f"  main thread alive — tick {i}")
        time.sleep(0.5)

    # Give the daemon threads time to finish speaking
    time.sleep(4)
    print("=== Self-test complete ===")
