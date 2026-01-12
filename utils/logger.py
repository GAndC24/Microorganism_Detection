import os
from datetime import datetime
import json


class Logger:
    def __init__(
        self,
        model_name : str,
        trainer_config : dict = None,      # training configuration
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

            # create train_config.json
            self.config_file_path = os.path.join(self.log_dir, "train_config.json")
            with open(self.config_file_path, 'w') as config_file:
                 json.dump(trainer_config, config_file, indent=4)  # Write trainer_config as JSON

            # create training_metrics.jsonl
            self.metrics_file_path = os.path.join(self.log_dir, "training_metrics.jsonl")
            with open(self.metrics_file_path, 'w') as metrics_file:
                metrics_file.write("")
        else:       # continue existing logs
            self.log_dir = os.path.join(root, continue_existing)
            self.log_file_path = os.path.join(self.log_dir, "train_log.txt")
            with open(self.log_file_path, 'a') as log_file:
                log_file.write(f"--------------------Continue train {model_name}-------------------\n"
                               f"Continue Time: {datetime.now()}\n"
                               "\n")
            self.config_file_path = os.path.join(self.log_dir, "train_config.json")
            self.metrics_file_path = os.path.join(self.log_dir, "training_metrics.jsonl")

    def get_log_dir(self)-> str:
        return self.log_dir

    def add_info(self, info : str)-> None:
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"{info}")
        print(f"Log info: {info}")

    def add_metrics(self, metrics : dict)-> None:
        with open(self.metrics_file_path, 'a') as metrics_file:
            json.dump(metrics, metrics_file)
            metrics_file.write("\n")
        print("Train metrics:\n")
        for k, v in metrics.items():
            print(f"{k}: {v}")
