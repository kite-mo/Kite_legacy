from exceptions.logic import LogicException


class MutationException(LogicException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ModelingException(LogicException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class PredictionException(LogicException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ScoringException(LogicException):
    def __init__(self, message: str) -> None:
        super().__init__(message)

