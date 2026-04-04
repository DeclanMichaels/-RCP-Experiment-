#!/usr/bin/env python3
"""
Run all unit tests and display results.

Usage:
    python run_tests.py          # Run all tests
    python run_tests.py -v       # Verbose (show each test name)
"""

import subprocess
import sys


def main():
    verbose = "-v" in sys.argv

    test_files = [
        ("collect.py functions", "test_collect.py"),
        ("analyze.py functions", "analysis/test_analyze.py"),
        ("permutation_tests.py functions", "analysis/test_permutation.py"),
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    print("=" * 60)
    print("RCP EXPERIMENT -- UNIT TEST RUNNER")
    print("=" * 60)

    for label, test_file in test_files:
        print(f"\n--- {label} ({test_file}) ---")

        args = [sys.executable, "-m", "pytest", test_file, "-q", "--tb=short",
                "-W", "ignore::FutureWarning"]
        if verbose:
            args = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short",
                    "-W", "ignore::FutureWarning"]

        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            # Filter out FutureWarnings from sklearn
            lines = [l for l in result.stderr.splitlines()
                     if "FutureWarning" not in l
                     and "warnings.warn" not in l
                     and "will be dropped" not in l
                     and "will change" not in l
                     and "deprecated" not in l]
            if lines:
                print("\n".join(lines))

        # Parse counts from pytest output
        for line in result.stdout.splitlines():
            if "passed" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "passed" and i > 0:
                        try:
                            total_passed += int(parts[i - 1])
                        except ValueError:
                            pass
                    if p == "failed" and i > 0:
                        try:
                            count = int(parts[i - 1].rstrip(","))
                            total_failed += count
                        except ValueError:
                            pass

        if result.returncode != 0:
            failures.append(label)

    # Summary
    print("\n" + "=" * 60)
    print("UNIT TEST SUMMARY")
    print("=" * 60)
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Total:  {total_passed + total_failed}")

    if failures:
        print(f"\n  FAILURES in: {', '.join(failures)}")
        print("\n  Fix failing tests before running the experiment.")
        sys.exit(1)
    else:
        print(f"\n  ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
