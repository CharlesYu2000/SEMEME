import logging
import sys
import os
from datetime import datetime

def setup_logger(log_dir="logs", log_prefix="sememe", logger_name=None):
    if logger_name is None:
        logger_name = log_prefix
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_prefix}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

        fileHandler = logging.FileHandler(log_path)
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        logger.addHandler(consoleHandler)

    return logger


def add_run_file_handler(base_dir, model_name, method, num_remove_blocks, nsamples,
                         seed=0, heat_size=None, dataset=None, logger_name="sememe"):
    logger = logging.getLogger(logger_name)

    for h in [h for h in logger.handlers if getattr(h, "_sememe_run_handler", False)]:
        logger.removeHandler(h)
        h.close()

    model_tag = str(model_name).rstrip("/").split("/")[-1]
    parts = [str(num_remove_blocks), str(nsamples), f"seed{seed}"]
    if heat_size is not None:
        parts.append(f"heat{heat_size}")
    if dataset:
        parts.append(str(dataset))
    run_dir = os.path.join(base_dir, model_tag, str(method))
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "_".join(parts) + ".log")

    formatter = next((h.formatter for h in logger.handlers if h.formatter is not None), None)
    if formatter is None:
        formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    fh._sememe_run_handler = True
    logger.addHandler(fh)
    logger.info(f"Per-run log -> {log_path}")
    return fh, log_path
