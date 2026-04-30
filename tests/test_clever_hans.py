"""
tests/test_clever_hans.py — Track C (Riya Bhart)
Tests for AttributionCollector and CleverHansAuditor.

Run:
    cd visionvoice
    python tests/test_clever_hans.py
"""

import sys
import os
import json
import numpy as np
import tempfile
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from clever_hans import AttributionCollector, CleverHansAuditor, NavigationDecision


def make_random_map(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((224, 224)).astype(np.float32)


def make_random_frame() -> Image.Image:
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ── Collector Tests ───────────────────────────────────────────────────────────

def test_01_import():
    print("\n[test 01] Import clever_hans module ...")
    from clever_hans import AttributionCollector, CleverHansAuditor
    print("  ✓ Import OK")


def test_02_collector_record_and_len():
    print("\n[test 02] AttributionCollector.record() and __len__ ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = AttributionCollector(save_dir=tmpdir)
        for i in range(5):
            collector.record(
                pil_frame=make_random_frame(),
                action="MoveAhead",
                attnlrp_map=make_random_map(i),
                step=i,
                surprise_score=float(i) * 0.1,
            )
        assert len(collector) == 5, f"Expected 5 records, got {len(collector)}"
        print(f"  ✓ Recorded 5 decisions")


def test_03_collector_saves_files():
    print("\n[test 03] Collector saves .npy and .png files ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = AttributionCollector(save_dir=tmpdir)
        collector.record(make_random_frame(), "RotateLeft", make_random_map(0))
        collector.record(make_random_frame(), "MoveAhead", make_random_map(1))

        npy_files = list(Path(tmpdir).glob("attnlrp_*.npy"))
        png_files = list((Path(tmpdir) / "frames").glob("frame_*.png"))
        assert len(npy_files) == 2, f"Expected 2 .npy files, got {len(npy_files)}"
        assert len(png_files) == 2, f"Expected 2 .png files, got {len(png_files)}"

        # Verify .npy content
        loaded = np.load(str(npy_files[0]))
        assert loaded.shape == (224, 224)
        print(f"  ✓ Saved {len(npy_files)} .npy and {len(png_files)} .png files")


def test_04_collector_manifest():
    print("\n[test 04] Collector saves manifest.json ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = AttributionCollector(save_dir=tmpdir)
        actions = ["MoveAhead", "RotateLeft", "MoveAhead"]
        for i, action in enumerate(actions):
            collector.record(make_random_frame(), action, make_random_map(i))
        manifest_path = collector.save_manifest()
        assert os.path.exists(manifest_path)

        with open(manifest_path) as f:
            data = json.load(f)
        assert len(data) == 3
        assert data[1]["action"] == "RotateLeft"
        print(f"  ✓ Manifest saved with {len(data)} entries")


# ── Auditor Tests ─────────────────────────────────────────────────────────────

def test_05_auditor_load_from_dir():
    print("\n[test 05] CleverHansAuditor.load_from_dir() ...")
    with tempfile.TemporaryDirectory() as maps_dir, tempfile.TemporaryDirectory() as report_dir:
        # Save some .npy maps manually
        for i in range(10):
            np.save(os.path.join(maps_dir, f"attnlrp_{i:04d}.npy"), make_random_map(i))

        auditor = CleverHansAuditor(maps_dir=maps_dir, report_dir=report_dir)
        n = auditor.load_from_dir()
        assert n == 10, f"Expected 10, got {n}"
        print(f"  ✓ Loaded {n} maps from directory scan")


def test_06_auditor_load_from_manifest():
    print("\n[test 06] CleverHansAuditor.load_from_manifest() ...")
    with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as report_dir:
        collector = AttributionCollector(save_dir=tmpdir)
        for i in range(15):
            collector.record(
                make_random_frame(), "MoveAhead" if i % 2 == 0 else "RotateLeft",
                make_random_map(i), step=i
            )
        manifest_path = collector.save_manifest()

        auditor = CleverHansAuditor(maps_dir=tmpdir, report_dir=report_dir)
        n = auditor.load_from_manifest(manifest_path)
        assert n == 15
        print(f"  ✓ Loaded {n} samples from manifest")


def test_07_auditor_insufficient_samples():
    print("\n[test 07] Audit with too few samples returns error ...")
    with tempfile.TemporaryDirectory() as maps_dir, tempfile.TemporaryDirectory() as report_dir:
        for i in range(5):
            np.save(os.path.join(maps_dir, f"attnlrp_{i:04d}.npy"), make_random_map(i))

        auditor = CleverHansAuditor(maps_dir=maps_dir, report_dir=report_dir, min_samples=30)
        auditor.load_from_dir()
        result = auditor.run_audit()
        assert "error" in result and result["error"] == "insufficient_samples"
        print(f"  ✓ Correctly refused audit with {result['n_samples']} samples (need ≥30)")


def test_08_auditor_full_run():
    print("\n[test 08] Full CleverHansAuditor run with 50 synthetic maps ...")
    with tempfile.TemporaryDirectory() as maps_dir, tempfile.TemporaryDirectory() as report_dir:
        collector = AttributionCollector(save_dir=maps_dir)
        actions = ["MoveAhead", "RotateLeft", "RotateRight", "MoveAhead", "LookUp"]
        for i in range(50):
            # Create maps with 3 distinct patterns to encourage cluster separation
            if i % 3 == 0:
                m = np.zeros((224, 224), dtype=np.float32)
                m[:80, :] = 1.0  # top stripe
            elif i % 3 == 1:
                m = np.zeros((224, 224), dtype=np.float32)
                m[:, :80] = 1.0  # left stripe
            else:
                m = np.zeros((224, 224), dtype=np.float32)
                m[80:160, 80:160] = 1.0  # center block
            # Add small noise
            m += np.random.rand(224, 224).astype(np.float32) * 0.05

            collector.record(
                make_random_frame(),
                actions[i % len(actions)],
                m,
                step=i,
            )
        manifest_path = collector.save_manifest()

        auditor = CleverHansAuditor(
            maps_dir=maps_dir,
            report_dir=report_dir,
            k_range=(2, 3, 4),
            min_samples=20,
        )
        auditor.load_from_manifest(manifest_path)
        result = auditor.run_audit()

        assert "error" not in result
        assert "best_k" in result
        assert "silhouette_score" in result
        assert os.path.exists(result["report_path"])

        print(f"  Best k          : {result['best_k']}")
        print(f"  Silhouette score: {result['silhouette_score']:.4f}")
        print(f"  Report path     : {result['report_path']}")

        # With 3 clearly distinct patterns, silhouette should be decent
        assert result["silhouette_score"] > 0.20, (
            f"Silhouette {result['silhouette_score']:.4f} too low for clearly separated maps"
        )
        print("  ✓ Full audit run complete")


def test_09_cluster_pngs_saved():
    print("\n[test 09] Cluster visualisation PNGs are saved ...")
    with tempfile.TemporaryDirectory() as maps_dir, tempfile.TemporaryDirectory() as report_dir:
        collector = AttributionCollector(save_dir=maps_dir)
        for i in range(40):
            m = np.random.rand(224, 224).astype(np.float32)
            collector.record(make_random_frame(), "MoveAhead", m)
        manifest_path = collector.save_manifest()

        auditor = CleverHansAuditor(maps_dir=maps_dir, report_dir=report_dir, min_samples=30)
        auditor.load_from_manifest(manifest_path)
        result = auditor.run_audit()

        k = result["best_k"]
        for i in range(k):
            png = Path(report_dir) / f"cluster_{i}_attnlrp.png"
            assert png.exists(), f"Missing cluster PNG: {png}"
        print(f"  ✓ {k} cluster PNG(s) saved to {report_dir}")


def run_all_tests():
    print("=" * 60)
    print("CLEVER HANS TESTS — Vision-to-Voice Track C")
    print("=" * 60)

    tests = [
        test_01_import,
        test_02_collector_record_and_len,
        test_03_collector_saves_files,
        test_04_collector_manifest,
        test_05_auditor_load_from_dir,
        test_06_auditor_load_from_manifest,
        test_07_auditor_insufficient_samples,
        test_08_auditor_full_run,
        test_09_cluster_pngs_saved,
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
            import traceback; traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
