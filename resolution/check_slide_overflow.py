#!/usr/bin/env python3
"""Flag slides whose content is clipped by the bottom (or right) page edge.

Marp does not error when a slide overflows -- it silently crops. Character
counts are a poor proxy (a 965-char slide fit while a 923-char one did not,
because the two differ in structure, not in length), so this measures the
rendered pixels instead.

A page is flagged when non-background pixels appear inside a thin band at the
very edge of the page: text that ends cleanly always leaves the margin blank,
whereas clipped text is cut mid-glyph and touches the boundary.

usage: python check_slide_overflow.py <deck.pdf> [<deck.pdf> ...]
"""
import glob
import os
import subprocess
import sys
import tempfile

import matplotlib.image as mpimg
import numpy as np

BAND = 6          # px band at the page edge to inspect
DARK = 0.80       # below this luminance counts as ink (bg is white)
MINPIX = 25       # ink pixels in the band before we call it clipped
DPI = 70


def page_images(pdf, tmp):
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", pdf,
                    os.path.join(tmp, "p")], check=True)
    return sorted(glob.glob(os.path.join(tmp, "p-*.png")))


def check(pdf):
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        pages = page_images(pdf, tmp)
        for i, f in enumerate(pages, 1):
            im = mpimg.imread(f)
            lum = im[..., :3].mean(axis=2) if im.ndim == 3 else im
            bottom = lum[-BAND:, :]
            # the page number lives near the right edge; ignore the far right
            right = lum[:, -BAND:][: int(0.90 * lum.shape[0])]
            nb = int((bottom < DARK).sum())
            nr = int((right < DARK).sum())
            # A section divider is a full-bleed red page, so its entire edge
            # band is "ink". That is a background fill, not clipped text --
            # clipped glyphs leave most of the band blank. Skip near-uniform
            # bands.
            if (bottom < DARK).mean() > 0.9 or (right < DARK).mean() > 0.9:
                continue
            if nb > MINPIX or nr > MINPIX:
                bad.append((i, nb, nr))
    return len(pages), bad


def main():
    rc = 0
    for pdf in sys.argv[1:]:
        n, bad = check(pdf)
        print(f"{os.path.basename(pdf)}  ({n} pages)")
        if not bad:
            print("   no clipped pages")
            continue
        rc = 1
        for i, nb, nr in bad:
            where = []
            if nb > MINPIX:
                where.append(f"bottom({nb}px)")
            if nr > MINPIX:
                where.append(f"right({nr}px)")
            print(f"   page {i:3d}  CLIPPED at {', '.join(where)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
