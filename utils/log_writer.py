from glob import glob
import os
from datetime import datetime 

class Log_writer(): 

    def __init__(self, output_directory : str, total_epochs : int): 
        self.output_dir = output_directory 
        self.total_epochs = total_epochs
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure the directory exists
        log_files_count = len(glob(os.path.join(output_directory, "*.txt")))
        self.output_file_dir = os.path.join(self.output_dir, f"Log_{log_files_count}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

    def write(self, epoch, **kwargs):
        """
        Documents losses and other values for every epoch of training.
        """
        with open(self.output_file_dir, 'a') as writer: 
            log = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{epoch}/{self.total_epochs}] "
            for key, value in kwargs.items(): 
                log += f"| {key}: {round(value, 3)} | "
            
            writer.write(log + "\n")
            writer.flush()

    def log_error(self, error_message):
        """
        Documents specifically errors or crashes during long training protocols
        """
        with open(self.output_file_dir, 'a') as writer:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log = f"[ERROR] [{timestamp}] {error_message}\n"
            writer.write(log)
