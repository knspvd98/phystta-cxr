import logging
from pathlib import Path
from typing import Optional


def setup_logging(output_dir: Path, level: str = "INFO", name: str = "phystta", enable_wandb: bool = False, wandb_project: str = "phystta", wandb_entity: Optional[str] = None, config: Optional[dict] = None) -> logging.Logger:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(output_dir / "experiment.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if enable_wandb:
        try:
            import wandb
            wandb.init(project=wandb_project, entity=wandb_entity, config=config or {})
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")

    return logger
