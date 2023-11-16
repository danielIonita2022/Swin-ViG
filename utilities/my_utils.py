import os
import shutil

import psutil
import datetime
import time


def split_folder(original_folder, new_folder1, new_folder2, split_count):
    # Get a list of files in the original folder
    files = [f for f in os.listdir(original_folder) if os.path.isfile(os.path.join(original_folder, f))]

    # Create new folders if they don't exist
    os.makedirs(new_folder1, exist_ok=True)
    os.makedirs(new_folder2, exist_ok=True)

    # Split files and move them
    for i, file in enumerate(files):
        if i < split_count:
            shutil.copy2(os.path.join(original_folder, file), os.path.join(new_folder1, file))
        else:
            shutil.copy2(os.path.join(original_folder, file), os.path.join(new_folder2, file))
def log_memory_usage(log_file):
    with open(log_file, "a") as file:
        while True:
            # Get memory usage
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Format the timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Create the log message
            log_message = f"{timestamp} - RAM: {memory.percent}% used, Swap: {swap.percent}% used\n"

            # Write to log file
            file.write(log_message)
            file.flush()

            # Log every 10 seconds (or choose your own interval)
            time.sleep(2)
