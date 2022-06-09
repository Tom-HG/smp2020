import logging
import time
import os


def get_logger(log_folder, to_console=True, to_file=True):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] ## %(message)s')
    timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())

    # 输出到文件
    if to_file:
        file_handler = logging.FileHandler(os.path.join(log_folder, '{}.txt'.format(timestamp)))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 输出到控制台
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # logging.info(str(args))
    return logger
