from glob import glob
import os
from datetime import datetime 

class Log_writer(): 

    def __init__(self, output_directory : str, total_epochs : int): 
        self.output_dir = output_directory 
        self.total_epochs = total_epochs
        self.output_file_dir = os.path.join(self.output_dir, f"Log_{len(glob(output_directory + "/*.txt"))}_{datetime.now().strftime("%Y%m/%d_%H:%M:%S")}.txt")

    def write(self, epoch, **kwargs):
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
