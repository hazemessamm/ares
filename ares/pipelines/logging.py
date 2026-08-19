import logging
import os

logger = logging.getLogger("ares")


def setup_logging(
    log_dir: str,
    run_name: str,
    resume: bool = False,
    console_only: bool = False,
):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    root = logging.getLogger("ares")
    root.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if not console_only:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{run_name}.log"),
            mode="a" if resume else "w",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
