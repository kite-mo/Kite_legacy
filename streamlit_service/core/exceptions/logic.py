class LogicException(Exception):
    def __init__(self, message: str) -> None:
        self.name = type(self).__name__
        self.message = message
