"""Compile selected original kernels with the locked ROCm 10 toolchain."""

import argparse

from .loader import KERNELS, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="*", choices=list(KERNELS))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    for name in args.modules or KERNELS:
        lib = load(name, verbose=args.verbose)
        print(f"{name}: {lib.__file__}", flush=True)


if __name__ == "__main__":
    main()
