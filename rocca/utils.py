from icecream import ic


def log_msg(log, msg: str):
    """Log and pretty-print the given message

    `log` is assumed to be one of the OpenCog loggers.
    """
    log.info(msg)
    ic(msg)
