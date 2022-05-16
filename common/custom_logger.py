import logging
from prettytable import PrettyTable


class CustomFormatter(logging.Formatter):
    green = "\x1b[32;20m"
    blue = "\x1b[36;20m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(name)s] [%(levelname)s]: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomPrettyTable(PrettyTable):
    def add_row(self, row, status: bool = True) -> None:
        custom_formetter = CustomFormatter()
        if status:
            return super().add_row(row)
        else:
            row = list(map(lambda x: custom_formetter.bold_red + str(x) + custom_formetter.reset, row))
            return super().add_row(row)


def CustomLogger(name="customLogger", level=logging.INFO):
    root_logger = logging.getLogger()

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if root_logger.hasHandlers():
        root_logger.handlers = []

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())
    root_logger.addHandler(ch)

    return logger
