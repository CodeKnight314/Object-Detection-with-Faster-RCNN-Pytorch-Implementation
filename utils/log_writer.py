from glob import glob
import os
from datetime import datetime 

class LOGWRITER(): 

    def __init__(self, output_directory : str, total_epochs : int): 
        """
        Initializes the LOGWRITER with a directory for output and the expected total number of epochs. It creates the 
        output directory if it does not already exist, and sets up a new log file with a unique name based on a timestamp
        and a count of existing log files.
        """
        self.output_dir = output_directory 
        self.total_epochs = total_epochs
        os.makedirs(self.output_dir, exist_ok=True)
        log_files_count = len(glob(os.path.join(output_directory, "*.txt")))
        self.output_file_dir = os.path.join(self.output_dir, f"Log_{log_files_count}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

    def write(self, epoch, **kwargs):
        """
        Logs the specified epoch's metrics and other key-value paired information to the log file.

        Args:
            epoch (int): The current epoch number during training.
            **kwargs: Arbitrary keyword arguments representing various metrics (like loss, accuracy, etc.) to log.
        """
        with open(self.output_file_dir, 'a') as writer: 
            log = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{epoch}/{self.total_epochs}] "
            for key, value in kwargs.items(): 
                log += f"| {key}: {round(value, 3)} | "
            
            writer.write(log + "\n")
            writer.flush()

    def log_error(self, error_message):
        """
        Logs an error message with a timestamp to the log file, useful for recording exceptions or problems encountered
        during training.

        Args:
            error_message (str): The error message to log.
        """
        with open(self.output_file_dir, 'a') as writer:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log = f"[ERROR] [{timestamp}] {error_message}\n"
            writer.write(log)