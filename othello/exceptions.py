class OthelloRuntimeException(Exception):
    def __init__(self, message):
        self.message = message


class IllegalShapeException(OthelloRuntimeException):
    def __init__(self, expected, actual):
        super().__init__("expected shape: {}, actual shape: {}".format(expected, actual))


class IllegalIndexException(OthelloRuntimeException):
    def __init__(self, expected, actual):
        super().__init__("shape: {}, index access: {}".format(expected, actual))
