import os
import yaml
import logging
from logging import Logger



class Config:
    """Class to handle paths and configurations for the framework."""



    def __init__(self) -> None:
        """Initializes the configuration by setting up paths and loading config files."""

        self.normal_or_hpc = "normal"
        self.name_dir_exp = "project_dir"
        self.username = "mtritto0"
        self.SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.exp_name =  os.environ["EXP_NAME"]

        if self.normal_or_hpc == "normal":
            self.project_dir = f"/your/absolute_path/to/{self.name_dir_exp}/GradeSQL"
        else:
            self.project_dir = f"/your/absolute_path/to/{self.username}/{self.name_dir_exp}/GradeSQL"

        desired_dir = self.project_dir + "/src/"
        os.chdir(desired_dir)
        
        CONFIG_PATH = os.path.join(self.SCRIPT_DIR, "..", "..", "recipes", f"{self.exp_name}.yaml")

        with open(CONFIG_PATH, 'r') as config_file:
            config = yaml.safe_load(config_file)
            self.config = config
        


    def get_config(self):
        """Returns the loaded configuration."""
        return self.config
    


    def get_exp_name(self):
        """Returns the experiment name."""
        return self.exp_name
    

    
    def construct_path(self, path: str) -> str:
        """Constructs a full path from the script directory."""
        return os.path.join(self.SCRIPT_DIR, path)



def setup_logging() -> Logger:
    """Sets up logging for the experiment, appending to an existing log file."""
        
    conf = Config()
    exp_name = conf.get_exp_name()
    logging_file_name = conf.construct_path(f"../../results/{exp_name}/logger.log")
    
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=logging_file_name,
        filemode='a'
    )
    return logging.getLogger(__name__)



def setup_logging_erase() -> Logger:
    """Sets up logging for the experiment, erasing previous log files if they exist."""
        
    conf = Config()
    exp_name = conf.get_exp_name()
    logging_file_name = conf.construct_path(f"../../results/{exp_name}/logger.log")

    log_dir = os.path.dirname(logging_file_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    if os.path.exists(logging_file_name):
        os.remove(logging_file_name)
    
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=logging_file_name,
        filemode='w'
    )
    return logging.getLogger(__name__)