import logging
from pathlib import Path
import wandb

## Functional logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(levelname)s: %(name)s %(message)s. Created at: %(created)f.')
# file_handler = logging.FileHandler('./project.log')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

class CustomLogger(logging.Logger):
    """Simple python class to test logging"""
    def __init__(self, filename, level=logging.INFO):
        super().__init__(name=__name__, level=level)
        self.filename = filename
        self.format = "%(levelname)s: %(message)s"

        # Set up the file handler
        file = Path(self.filename)
        file.touch(exist_ok=True)
        file_handler = logging.FileHandler(filename=file)
        file_handler.setFormatter(logging.Formatter(self.format))
        self.addHandler(file_handler)

