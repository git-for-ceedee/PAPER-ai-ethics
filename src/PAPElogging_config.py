# --------------------------------
# Pick-a-Path AI Ethics
# logging setup
# --------------------------------

import logging, sys
from pathlib import Path

def setup_logging(level: int = logging.INFO, log_to_file: bool = False, log_dir: Path | None = None):
    # Clear existing handlers (prevents duplicate logs if run multiple times)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_to_file and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "run.log", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
   

