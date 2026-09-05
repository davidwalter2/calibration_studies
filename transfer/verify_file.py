#!/usr/bin/env python3
"""Verify a locally copied file against the size and adler32 recorded in DAS.

The xrdcp exit code is NOT sufficient evidence that a copy succeeded: with
`--retry` it can leave a short file behind and still exit 0 (see the project
note on atomic publishing).  Every file therefore gets an independent size
check plus a streaming adler32 over the bytes actually on disk.

Exit 0 = verified, 1 = mismatch/missing.  Prints "<actual_adler32> <bytes>".
"""
import argparse
import os
import sys
import zlib

CHUNK = 16 << 20  # 16 MiB


def adler32(path, chunk=CHUNK):
    val = 1
    with open(path, "rb", buffering=0) as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            val = zlib.adler32(buf, val)
    return val & 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("size", type=int)
    ap.add_argument("adler32", help="expected adler32, hex (DAS convention)")
    ap.add_argument("--size-only", action="store_true",
                    help="check size only (fast pre-screen)")
    args = ap.parse_args()

    if not os.path.isfile(args.path):
        print("MISSING", file=sys.stderr)
        return 1
    actual_size = os.path.getsize(args.path)
    if actual_size != args.size:
        print("SIZE_MISMATCH got=%d want=%d" % (actual_size, args.size), file=sys.stderr)
        return 1
    if args.size_only:
        print("size-ok %d" % actual_size)
        return 0

    # DAS stores adler32 as hex, sometimes without leading zeros.
    want = int(args.adler32, 16)
    got = adler32(args.path)
    if got != want:
        print("ADLER_MISMATCH got=%08x want=%08x" % (got, want), file=sys.stderr)
        return 1
    print("%08x %d" % (got, actual_size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
