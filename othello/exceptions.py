class IllegalShapeException(Exception):
    def __init__(self, expected, actual):
        self.message = "expected shape: {}, actual shape: {}".format(expected, actual)


class IllegalIndexException(Exception):
    def __init__(self, expected, actual):
        self.message = "shape: {}, index access: {}".format(expected, actual)


class OthelloRuntimeException(Exception):
    def __init__(self, message):
        self.message = message
