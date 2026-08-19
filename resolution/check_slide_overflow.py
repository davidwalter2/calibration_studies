#!/usr/bin/env python3
"""Flag slides whose content is clipped by the bottom (or right) page edge.

Marp does not error when a slide overflows -- it silently crops. Character
counts are a poor proxy (a 965-char slide fit while a 923-char one did not,
because the two differ in structure, not in length), so this measures the
rendered pixels instead.

TWO detectors, because each misses what the other catches.

  EDGE BAND -- ink inside a thin band at the very edge of the page. Text that
  ends cleanly always leaves the margin blank, whereas clipped text is cut
  mid-glyph and touches the boundary.

  BELOW PAGINATION -- ink from the body that lies below the page number, in a
  bottom-most row wide enough to be a severed table rule or line of text.
  This exists because the edge band alone gave a FALSE PASS on a real deck:
  when Marp pushes a whole table ROW off the canvas, that row's ink is simply
  gone, so the page comes out with a LARGER bottom margin (10 px at DPI 70)
  and an entirely EMPTY edge band, and the band test reports the page clean
  while the last row of the table is missing from the talk.

Both were calibrated on a fixture deck holding the two slides that were
actually clipped in 260819_cleanprop_walter and five that were not
(`cleanprop/slides/overflow_fixture.md` -- keep it building). At DPI 70,
body = left 90 % of the width, pagination = bottom-right corner box:

    page                                below page num   bottom row width
    BAD  ionization, table row lost         +12 px            30.4 %
    BAD  real geometry, cut mid-glyph       +22 px             6.4 %
    GOOD low footnote                       -64 px             1.8 %
    GOOD plots caption to the last line      +8 px             0.8 %
    GOOD roomy                             -157 px             1.3 %
    GOOD ends with a table                 -184 px            18.5 %

Neither column separates the classes on its own, so the test is the
CONJUNCTION: a clean slide that ENDS in a table has an 18.5 % wide bottom row
but sits far ABOVE the page number, and a clean slide with a footnote sits
below the page number but ends on a narrow line of text.

MINROW = 10 % is set high on purpose. The only class this detector has to
catch by itself is a severed table RULE (30.4 %); a page cut mid-glyph trips
the edge band anyway (73 px against MINPIX = 25), so nothing is lost by
letting the 6.4 % page fall to the other detector. A first attempt at 2.5 %
was calibrated on the fixture alone and then FALSE-POSITIVED on
260814_cvh_noise_model_walter page 21 -- an ordinary last bullet line, fully
visible, whose baseline row is 2.7 % wide. 10 % sits 3.7x above that and 3.0x
below the rule it must catch; 2.5 % had only 1.5x on either side.

The width statistic is resolution dependent (a 1 px CSS rule antialiases away
above ~110 dpi: the ionization page reads 30.4 % at DPI 70 and 4.2 % at 150),
so DPI is part of the calibration and must not be raised without re-measuring.
A deck with `paginate: false` has no page number to compare against; there the
second detector is skipped rather than guessed, and the page is reported as
band-only.

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
DPI = 70          # part of the MINROW calibration -- see the module docstring

BODY = 0.90       # fraction of the width that is body, not pagination
PAGCOL = 0.92     # pagination lives to the right of this fraction ...
PAGROW = 0.85     # ... and below this fraction of the height
MINROW = 0.10     # bottom body row must span this much of the body width


def page_images(pdf, tmp):
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", pdf,
                    os.path.join(tmp, "p")], check=True)
    return sorted(glob.glob(os.path.join(tmp, "p-*.png")))


def below_pagination(ink):
    """(rows the body reaches past the page number, width of its lowest row).

    Returns (None, None) when the page has no pagination number to compare
    against, so the caller skips this detector instead of guessing.
    """
    h, w = ink.shape
    body = ink[:, : int(BODY * w)]
    pag = ink[int(PAGROW * h):, int(PAGCOL * w):]
    if not pag.any() or not body.any():
        return None, None
    pbase = int(PAGROW * h) + int(np.where(pag.any(axis=1))[0].max())
    blast = int(np.where(body.any(axis=1))[0].max())
    return blast - pbase, float(body[blast].mean())


def check(pdf):
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        pages = page_images(pdf, tmp)
        for i, f in enumerate(pages, 1):
            im = mpimg.imread(f)
            lum = im[..., :3].mean(axis=2) if im.ndim == 3 else im
            ink = lum < DARK
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
            over, roww = below_pagination(ink)
            nover = (over is not None and over > 0 and roww > MINROW)
            if nb > MINPIX or nr > MINPIX or nover:
                bad.append((i, nb, nr, over, roww))
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
        for i, nb, nr, over, roww in bad:
            where = []
            if nb > MINPIX:
                where.append(f"bottom band({nb}px)")
            if nr > MINPIX:
                where.append(f"right band({nr}px)")
            if over is not None and over > 0 and roww > MINROW:
                where.append(f"{over}px below the page number, "
                             f"bottom row {100 * roww:.1f}% wide")
            print(f"   page {i:3d}  CLIPPED -- {'; '.join(where)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
