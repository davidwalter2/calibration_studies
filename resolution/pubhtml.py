"""Housekeeping for plot directories served under ~/public_html.

Every output directory under ~/public_html wants the little PHP gallery
that renders the PDFs/PNGs it contains as a browsable index. Rather than
remembering to copy it by hand each time a new dated directory appears,
every plotting script calls `ensure_index(outdir)` right after it makes
the directory.

The canonical copy lives in ~/public_html/cvh/260814_cleanprop/index.php;
it is copied only if the target does not already have one (never
overwritten, so local tweaks survive) and only for directories that
actually sit under ~/public_html (a --outpath pointing somewhere else is
left alone).
"""

import os
import shutil

TEMPLATE = os.path.expanduser("~/public_html/cvh/260814_cleanprop/index.php")
_PUBHTML = os.path.realpath(os.path.expanduser("~/public_html"))


def ensure_index(outdir, template=TEMPLATE, logger=None):
    """Copy the plot-browser index.php into `outdir` unless it is there.

    Returns the path written, or None if nothing was done (already
    present, outside ~/public_html, or the template is missing).
    """
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
