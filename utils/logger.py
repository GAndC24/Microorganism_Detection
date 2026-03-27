import os
from datetime import datetime
import json
from typing import Dict, Any
import yaml
from pathlib import Path

class Logger:
    def __init__(
        self,
        model_name : str,
        config : Dict[str, Any],      # training configuration
        continue_existing: str = None,  # continue existing logs
        root : str = './logs'      # log root path
    )-> None:
        if continue_existing is None:       # new training
            # create log directory
            start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_dir = os.path.join(root, f"train_log_{start_time}")
            os.makedirs(self.log_dir, exist_ok=True)

            # create train_log.txt
            self.log_file_path = os.path.join(self.log_dir, "train_log.txt")
            current_time = datetime.now()
            with open(self.log_file_path, 'a') as log_file:
                log_file.write(f"--------------------Start train {model_name}-------------------\n"
                               f"Start Time: {current_time}\n"
                               "\n")

            # # create train_config.json
            # self.config_file_path = os.path.join(self.log_dir, "train_config.json")
            # with open(self.config_file_path, 'w') as config_file:
            #      json.dump(trainer_config, config_file, indent=4)  # Write trainer_config as JSON

            # create training_metrics.jsonl
            self.metrics_file_path = os.path.join(self.log_dir, "training_metrics.jsonl")
            with open(self.metrics_file_path, 'w') as metrics_file:
                metrics_file.write("")

            # save config as yaml
            self.config_file_path = os.path.join(self.log_dir, "config_ccam.yaml")
            self._save_yaml(config, self.config_file_path)

        else:       # continue existing logs
            self.log_dir = continue_existing
            self.log_file_path = os.path.join(self.log_dir, "train_log.txt")
            with open(self.log_file_path, 'a') as log_file:
                log_file.write(f"--------------------Continue train {model_name}-------------------\n"
                               f"Continue Time: {datetime.now()}\n"
                               "\n")
            self.metrics_file_path = os.path.join(self.log_dir, "training_metrics.jsonl")
            self.config_file_path = os.path.join(self.log_dir, "config_ccam.yaml")
            self._save_yaml(config, self.config_file_path)

    def get_log_dir(self)-> str:
        return self.log_dir

    def add_info(self, info : str)-> None:
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"{info}")
        print(f"\nLog info: {info}")

    def add_metrics(self, metrics : dict)-> None:
        with open(self.metrics_file_path, 'a') as metrics_file:
            json.dump(metrics, metrics_file)
            metrics_file.write("\n")
        print("\nTrain metrics:\n")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    def end_train(self)-> None:
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"--------------------End train-------------------\n"
                           f"End Time: {datetime.now()}\n"
                           "\n")

    def _save_yaml(self, data: Dict[str, Any], path: str) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

