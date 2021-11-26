import logging
import typing


def txrm_property(
    function: typing.Optional[typing.Callable] = None,
    fallback: typing.Any = None,
    **kwargs,
):
    class TxrmProperty:
        def __init__(self, fn, fallback):
            self.fn = fn
            self.fallback = fallback
            self.hidden_var_str = f"_{self.fn.__name__}"

        def __get__(self, obj, object_type=None):
            try:
                if getattr(obj, self.hidden_var_str, None) is None:
                    try:
                        setattr(obj, self.hidden_var_str, self.fn(obj))
                    except Exception:
                        if not obj.file_is_open:
                            raise IOError(
                                f"Cannot get {self.fn.__name__} while file is closed"
                            )
                        raise
            except Exception:
                if obj.strict:
                    raise
                logging.error(
                    "Failed to get property '%s'",
                    self.fn.__name__,
                    exc_info=True,
                )
            return getattr(obj, self.hidden_var_str, self.fallback)

        def __set__(self, obj, value):
            raise AttributeError(f"{self.fn.__name__} cannot be set.")

        def __delete__(self, obj):
            setattr(obj, self.hidden_var_str, None)

    if function:
        return TxrmProperty(function)
    else:

        def wrapper(function):
            return TxrmProperty(function, fallback, **kwargs)

        return wrapper
