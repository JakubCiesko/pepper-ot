from pathlib import Path

import yaml


class KnowledgeDistiller:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = yaml.safe_load(config_path)
