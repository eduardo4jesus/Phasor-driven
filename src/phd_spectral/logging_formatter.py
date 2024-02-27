import logging

from colorama import Fore, Style


class ColoredFormatter(logging.Formatter):
    _LEVEL_COLORS = {
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED,
    }

    def format(self, record) -> str:
        color = self._LEVEL_COLORS.get(record.levelno, Style.RESET_ALL)
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"


def set_logging_formatter() -> None:
    root_logger = logging.getLogger()
    default_handler = root_logger.handlers[0]
    default_handler.setFormatter(ColoredFormatter())
