class IllegalShapeException(Exception):
    def __init__(self, expected, actual):
        self.message = "expected shape: {}, actual shape: {}".format(expected, actual)
