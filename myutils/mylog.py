import logging
import os

def _setup_logger(log_dir="logs", log_file_name=".notionlogs", level=logging.INFO):
    """
    Sets up a Python logger for the application, ensuring a single log file
    for continuous writing. The logger can be retrieved from anywhere
    in the application using logging.getLogger('my_app_logger').

    Args:
        log_dir (str): Directory to store the log file. Created if it doesn't exist.
        log_file_name (str): The name of the log file. Logs will be appended to this file.
        level (int): Minimum logging level to record (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, log_file_name)

    # Get the logger instance with a unique name
    logger = logging.getLogger("my_logger")
    logger.setLevel(level)

    # Prevent adding handlers multiple times if the function is called more than once
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler (append mode 'a' for continuous writing)
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

Logger = _setup_logger()