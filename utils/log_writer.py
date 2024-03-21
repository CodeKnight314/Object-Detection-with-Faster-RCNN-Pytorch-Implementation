from glob import glob
import os
from datetime import datetime 

class Log_writer(): 

    def __init__(self, output_directory : str, total_epochs : int): 
        self.output_dir = output_directory 
        self.total_epochs = total_epochs
        self.output_file_dir = os.path.join(self.output_dir, f"Log_{len(glob(output_directory + "/*.txt"))}_{datetime.now().strftime("%Y%m/%d_%H:%M:%S")}.txt")

    def write(self, epoch, **kwargs):
        """
        Documents losses and other values for every epoch of training 

        Args: 
            epoch (int): the epoch that corresponds to the values for documentation
            **kwargs : loss variable name followed by value. Floats should be limited to 3 decimals
        """
        if os.path.exists(self.output_file_dir):
            with open(self.output_file_dir, 'a') as writer: 
                log = f"[{epoch}/{self.total_epochs}] "
                for key, value in kwargs.items(): 
                    log = log + f"| {key} : {value} | "
                
                writer.write(log)
                writer.close() 

        else: 
            with open(self.output_file_dir, 'x') as writer: 
                log = f"[{epoch}/{self.total_epochs}] "
                for key, value in kwargs.items(): 
                    log = log + f"| {key} : {value} | "
                writer.write(log)
                writer.close() 
