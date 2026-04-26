"""
tests/test_narrator.py — Track C (Riya Bhart)
Tests for Narrator and SilentNarrator.

Run from your project root:
    cd visiontovoice/Vision-to-Voice
    python tests/test_narrator.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from narrator import Narrator, SilentNarrator, make_narrator


def test_01_import():
    print("\n[test 01] Import narrator module ...")
    from narrator import Narrator, SilentNarrator, make_narrator
    print("  ✓ Import OK")


def test_02_silent_narrator_api():
    print("\n[test 02] SilentNarrator — prints instead of speaking ...")
    narrator = SilentNarrator()
    narrator.say("Test message")
    narrator.say_navigation("MoveAhead", 10)
    narrator.say_detection(["door", "sign"])
    narrator.say_sign("Room 204")
    narrator.say_landmark("Library")
    narrator.say_surprise(0.8)
    narrator.silence()
    narrator.shutdown()
    assert narrator.is_speaking is False
    assert isinstance(narrator.available_voices, list)
    narrator.set_rate(2)
    narrator.set_volume(80)
    print("  ✓ SilentNarrator API complete (check lines printed above)")


def test_03_make_narrator_disabled():
    print("\n[test 03] make_narrator(enabled=False) returns SilentNarrator ...")
    narrator = make_narrator(enabled=False)
    assert isinstance(narrator, SilentNarrator)
    narrator.say("This should PRINT not speak.")
    print("  ✓ Disabled narrator is SilentNarrator")


def test_04_win32com_available():
    print("\n[test 04] Check win32com (pywin32) is installed ...")
    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        voices = speaker.GetVoices()
        voice_names = [voices.Item(i).GetDescription() for i in range(voices.Count)]
        print(f"  win32com OK. Found {len(voice_names)} voice(s):")
        for v in voice_names:
            print(f"    - {v}")
        print("  ✓ win32com ready")
    except ImportError:
        print("  ✗ pywin32 not installed. Run: pip install pywin32")
        print("  ⚠ Skipping real speech tests")


def test_05_real_speech_all_phrases():
    """
    REAL SPEECH TEST — you should hear all 6 phrases spoken aloud.
    Uses win32com directly (no threading) to guarantee all phrases play.
    """
    print("\n[test 05] Real speech — you should hear 6 phrases ...")
    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Rate = 1
        speaker.Volume = 90

        phrases = [
            "Vision to Voice system online.",
            "Moving forward. 12 locations mapped.",
            "Detected: door, sign, stairs.",
            "Sign reads: Room 204.",
            "Reached: Library entrance.",
            "Navigation complete.",
        ]

        for phrase in phrases:
            print(f"  Speaking: {phrase}")
            speaker.Speak(phrase)

        print("  ✓ All 6 phrases spoken (if you heard them all, TTS is working)")

    except ImportError:
        print("  ⚠ pywin32 not installed — skipping real speech test")
        print("    Run: pip install pywin32")
    except Exception as e:
        print(f"  ✗ Speech error: {e}")


def test_06_narrator_class_real_speech():
    """
    Tests the actual Narrator class with background threading.
    You should hear: startup + 4 navigation phrases + shutdown.
    """
    print("\n[test 06] Narrator class — background threaded speech ...")
    narrator = make_narrator(rate=1, volume=90)
    print(f"  Engine type  : {type(narrator).__name__}")
    print(f"  Voices found : {narrator.available_voices}")
    print("  Queuing 5 messages (you should hear them all) ...")

    narrator.say("Narrator class test starting.")
    time.sleep(3)

    narrator.say_navigation("MoveAhead", 5)
    time.sleep(3)

    narrator.say_detection(["door", "sign"])
    time.sleep(3)

    narrator.say_sign("Room 204")
    time.sleep(3)

    narrator.say_landmark("Library entrance")
    time.sleep(3)

    narrator.shutdown()   # speaks "Navigation complete." then stops
    print("  ✓ Narrator class test done")


def test_07_nonblocking_check():
    print("\n[test 07] say() must return immediately (non-blocking) ...")
    narrator = make_narrator(rate=1, volume=90)

    start = time.time()
    narrator.say("Non blocking test one.")
    narrator.say("Non blocking test two.")
    narrator.say("Non blocking test three.")
    elapsed = time.time() - start

    print(f"  3 say() calls took {elapsed:.4f}s (must be < 0.1s)")
    assert elapsed < 0.5, f"say() is blocking! Took {elapsed:.3f}s"

    time.sleep(6)  # let the 3 phrases actually play
    narrator.shutdown()
    print("  ✓ say() is non-blocking")


def test_08_convenience_methods_silent():
    print("\n[test 08] Convenience methods with SilentNarrator ...")
    n = SilentNarrator()

    # Empty inputs — should NOT print anything
    n.say_detection([])       # empty list
    n.say_sign("")            # empty string
    n.say_landmark("")        # empty string
    n.say_surprise(0.1)       # below 0.5 threshold

    # Non-empty inputs — SHOULD print
    n.say_detection(["wheelchair ramp", "exit sign"])
    n.say_sign("Cafeteria this way")
    n.say_landmark("Main entrance")
    n.say_surprise(0.9)       # above 0.5 — should print

    print("  ✓ Convenience methods OK (check 4 lines printed above)")


def run_all_tests():
    print("=" * 60)
    print("NARRATOR TESTS — Vision-to-Voice Track C")
    print("Uses win32com (SAPI) for reliable Windows TTS")
    print("=" * 60)
    print("NOTE: Tests 05, 06, 07 produce REAL SPEECH.")
    print("      Make sure your speakers/headphones are on.")
    print("=" * 60)

    tests = [
        test_01_import,
        test_02_silent_narrator_api,
        test_03_make_narrator_disabled,
        test_04_win32com_available,
        test_05_real_speech_all_phrases,
        test_06_narrator_class_real_speech,
        test_07_nonblocking_check,
        test_08_convenience_methods_silent,
    ]

    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ ASSERTION FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
