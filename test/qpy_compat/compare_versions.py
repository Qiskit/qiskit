import argparse
import sys

from qiskit.qpy.interface import VERSION_PATTERN_REGEX


def main():
    parser = argparse.ArgumentParser(prog="compare_version", description="Compare version strings")
    parser.add_argument(
        "source_version", help="Source version of Qiskit that is generating the payload"
    )
    parser.add_argument("test_version", help="Version under test that will load the QPY payload")
    args = parser.parse_args()
    source_match = VERSION_PATTERN_REGEX.search(args.source_version)
    target_match = VERSION_PATTERN_REGEX.search(args.test_version)
    source_version = tuple(int(x) for x in source_match.group("release").split("."))
    target_version = tuple(int(x) for x in target_match.group("release").split("."))
    if source_version > target_version:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
