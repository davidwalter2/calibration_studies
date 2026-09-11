"""Housekeeping for plot directories served under ~/public_html.

Two jobs, both about the little PHP plot browser that renders a figure
directory in the web browser.

`savefig(fig, fn)` -- write a figure the way the browser can see it.
    The browser globs ONLY the raster images (`*.png`, `*.gif`) and hangs
    the `.pdf`/`.txt`/`.log` twins off the PNG's caption, so a directory
    that holds nothing but PDFs renders EMPTY.  Every `.pdf` therefore
    needs a `.png` next to it.  `savefig` does both in one call (and
    always with `bbox_inches="tight"`, per the user's standing rule), so
    no script can leave a PDF-only directory behind.

`ensure_index(outdir)` -- drop an `index.php` into a new directory.
    The current browser is installed once, globally, as
    ~/public_html/_index.php and wired up by the ~/public_html/.htaccess
    line

        DirectoryIndex index.html /~david_w/_index.php index.php

    which every subdirectory inherits, so a new plot directory is
    browsable without any file of its own -- and, because the shared
    browser is listed AHEAD of `index.php`, a per-directory copy is
    normally shadowed by it.  `ensure_index` is kept as the belt-and-
    braces fallback for directories served without that .htaccess (and
    for reading the directory locally): it copies the shared browser in
    as `index.php`, only if the target has none (never overwritten, so
    local tweaks survive) and only for directories that really sit under
    ~/public_html (a --outpath pointing elsewhere is left alone).

`TEMPLATE` is resolved at import rather than hard-coded, so moving the
browser does not break this module: the shared ~/public_html/_index.php if it
is there, else the newest plot-browser `index.php` under ~/public_html/cvh/*/.
"""

import glob
import os
import shutil

_PUBHTML = os.path.realpath(os.path.expanduser("~/public_html"))

#: the shared plot browser installed at the top of ~/public_html
_PRIMARY = os.path.expanduser("~/public_html/_index.php")

#: where to look for a copy if the shared browser is not at _PRIMARY
_FALLBACK_GLOB = os.path.expanduser("~/public_html/cvh/*/index.php")


def _is_browser(path):
    """True if `path` looks like the plot browser (globs images, links PDFs)."""
    try:
        with open(path, "r", errors="replace") as fh:
            txt = fh.read(20000)
    except OSError:
        return False
    return "glob(" in txt and ".pdf" in txt


def _find_template():
    if os.path.exists(_PRIMARY):
        return _PRIMARY
    cands = [p for p in glob.glob(_FALLBACK_GLOB) if _is_browser(p)]
    if cands:
        return max(cands, key=os.path.getmtime)
    return _PRIMARY          # missing; ensure_index() then warns and no-ops


TEMPLATE = _find_template()


def savefig(fig, fn, dpi=130, **kw):
    """Save `fig` to `fn`, plus a `.png` twin whenever `fn` is a `.pdf`.

    `bbox_inches="tight"` is applied unless the caller overrides it.  For a
    `.pdf` target `dpi` applies to the PNG twin only (the PDF is vector);
    for a raster target it applies to the file itself.  Returns the list of
    paths written.
    """
    fn = os.fspath(fn)
    kw.setdefault("bbox_inches", "tight")
    if fn.lower().endswith(".pdf"):
        fig.savefig(fn, **kw)
        png = fn[:-4] + ".png"
        fig.savefig(png, dpi=dpi, **kw)
        return [fn, png]
    fig.savefig(fn, dpi=dpi, **kw)
    return [fn]


def ensure_index(outdir, template=None, logger=None):
    """Copy the plot-browser index.php into `outdir` unless it is there.

    Returns the path written, or None if nothing was done (already
    present, outside ~/public_html, or the template is missing).
    """
    if template is None:
        template = TEMPLATE
    try:
        real = os.path.realpath(os.path.expanduser(outdir))
        if os.path.commonpath([real, _PUBHTML]) != _PUBHTML:
            return None
        dst = os.path.join(real, "index.php")
        if os.path.exists(dst):
            return None
        if not os.path.exists(template):
            if logger is not None:
                logger.warning(f"no index.php template at {template}")
            return None
        shutil.copyfile(template, dst)
        if logger is not None:
            logger.info(f"installed index.php in {real}")
        return dst
    except (OSError, ValueError) as exc:      # different drive, races, ...
        if logger is not None:
            logger.warning(f"could not install index.php in {outdir}: {exc}")
        return None
