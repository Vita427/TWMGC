import logging
import os
import sys


def metrics_info(metrics):
    return "metrics:\t%s" % '\t'.join(metrics)


def record_info(record):
    return '\t'.join(format(x, '.4f') for x in record)


class Logger(object):
    """Print a message and append it to ./log/<filename>.log at the same time."""

    LOG_DIR = './log'

    def __init__(self, filename):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        for handler in (logging.FileHandler(os.path.join(self.LOG_DIR, filename + ".log")),
                        logging.StreamHandler(sys.stdout)):
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        self.logger.critical(message)
        self._flush()
