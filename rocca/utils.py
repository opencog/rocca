from icecream import ic


def log_msg(log, msg: str):
    """

    `log` is assumed to be one of the OpenCog loggers.
    """
    log.fine(msg)
    ic(msg)