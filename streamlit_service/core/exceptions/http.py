class HttpException(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.name = type(self).__name__
        self.status_code = status_code
        self.message = message
