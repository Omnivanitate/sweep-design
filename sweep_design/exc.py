class NotEqualError(Exception):
    def __init__(self, *args: object) -> None:
        message = f"Different size x_array({args[0]}) and y_array({args[1]})"
        super().__init__(message)


class BadInputError(Exception):
    pass


class ConvertingError(Exception):
    def __init__(self, *args: object) -> None:
        message = f"Can not convert type {args[0]} into type {args[1]}."
        super().__init__(message)


class TypeFuncError(Exception):
    def __init__(self, *args: object) -> None:
        message = f'operation "{args[0]}" did not complete with type'
        '{args[1]} and type {args[2]}'
        super().__init__(message)
