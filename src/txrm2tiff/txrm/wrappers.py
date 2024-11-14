from __future__ import annotations
import logging
import typing

from ..utils.exceptions import TxrmFileError

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from .abc.file import TxrmFile
    from .abc.base import TxrmBase

    TxrmBaseType = typing.TypeVar("TxrmBaseType", bound=TxrmBase)
    TxrmFileType = typing.TypeVar("TxrmFileType", bound=TxrmFile)
    Param = typing.ParamSpec("Param")
    FallbackType = typing.TypeVar("FallbackType", typing.Any, None)
    RetType = typing.TypeVar("RetType", covariant=True)


class TxrmProperty(typing.Generic[RetType, FallbackType]):

    def __init__(
        self, fn: Callable[[TxrmFileType], RetType], fallback: FallbackType
    ) -> None:
        self.fn = fn
        self.fallback = fallback
        self.hidden_var_str = f"_{self.fn.__name__}"

    def __get__(
        self, obj: TxrmFile, object_type: type | None = None
    ) -> RetType | FallbackType:
        try:
            if self.hidden_var_str not in obj.__txrm_properties:
                obj.__txrm_properties[self.hidden_var_str] = self.fn(obj)  # type: ignore[arg-type]
        except Exception:
            if obj.strict:
                raise
            logging.error(
                "Failed to get property '%s'",
                self.fn.__name__,
                exc_info=True,
            )
        if self.hidden_var_str in obj.__txrm_properties:
            return typing.cast(RetType, obj.__txrm_properties[self.hidden_var_str])
        return self.fallback

    def __set__(self, obj: TxrmFile, value: typing.Any) -> typing.Never:
        raise AttributeError(f"{self.fn.__name__} cannot be set.")

    def __delete__(self, obj: TxrmFile) -> None:
        setattr(obj, self.hidden_var_str, None)


@typing.overload
def txrm_property(
    function: None = ...,
    fallback: FallbackType = ...,
) -> Callable[
    [Callable[[TxrmFileType], RetType]],
    TxrmProperty[RetType, FallbackType],
]: ...


@typing.overload
def txrm_property(
    function: Callable[[TxrmFileType], RetType],
    fallback: FallbackType = ...,
) -> TxrmProperty[RetType, FallbackType]: ...


def txrm_property(
    function: Callable[[TxrmFileType], RetType] | None = None,
    fallback: FallbackType = None,
) -> (
    TxrmProperty[RetType, FallbackType]
    | Callable[
        [
            Callable[
                [TxrmFileType],
                RetType,
            ]
        ],
        TxrmProperty[RetType, FallbackType],
    ]
):

    if function is None:

        def wrapper(
            function: Callable[[TxrmFileType], RetType]
        ) -> TxrmProperty[RetType, FallbackType]:
            return TxrmProperty(function, fallback)

        return wrapper

    return TxrmProperty(function, fallback)


@typing.overload
def uses_ole(
    function: None = ...,
    /,
    strict: typing.Literal[True] = ...,
) -> Callable[
    [Callable[typing.Concatenate[TxrmBaseType, Param], RetType]],
    Callable[typing.Concatenate[TxrmBaseType, Param], RetType],
]: ...


@typing.overload
def uses_ole(
    function: Callable[typing.Concatenate[TxrmBaseType, Param], RetType],
    /,
    strict: typing.Literal[True] = ...,
) -> Callable[typing.Concatenate[TxrmBaseType, Param], RetType]: ...


@typing.overload
def uses_ole(
    function: None = ...,
    /,
    strict: bool | None = ...,
) -> Callable[
    [Callable[typing.Concatenate[TxrmBaseType, Param], RetType]],
    Callable[typing.Concatenate[TxrmBaseType, Param], RetType | None],
]: ...


@typing.overload
def uses_ole(
    function: Callable[typing.Concatenate[TxrmBaseType, Param], RetType],
    /,
    strict: bool | None = ...,
) -> Callable[typing.Concatenate[TxrmBaseType, Param], RetType | None]: ...


def uses_ole(
    function: Callable[typing.Concatenate[TxrmBaseType, Param], RetType] | None = None,
    /,
    strict: bool | None = None,
) -> (
    Callable[typing.Concatenate[TxrmBaseType, Param], RetType | None]
    | Callable[
        [Callable[typing.Concatenate[TxrmBaseType, Param], RetType]],
        Callable[typing.Concatenate[TxrmBaseType, Param], RetType | None],
    ]
):
    _strict: bool | None = strict

    def wrapped_wrapper(
        function: Callable[typing.Concatenate[TxrmBaseType, Param], RetType],
    ) -> Callable[
        typing.Concatenate[TxrmBaseType, Param],
        RetType | None,
    ]:

        @typing.overload
        def wrapped(
            self: TxrmBaseType | None,
            /,
            *args: typing.Any,
            strict: typing.Literal[True] = ...,
            **kwargs: typing.Any,
        ) -> RetType: ...

        @typing.overload
        def wrapped(
            self: None,
            /,
            *args: typing.Any,
            strict: bool = ...,
            **kwargs: typing.Any,
        ) -> RetType | None: ...

        @typing.overload
        def wrapped(
            self: TxrmBaseType,
            /,
            *args: typing.Any,
            strict: bool | None = ...,
            **kwargs: typing.Any,
        ) -> RetType | None: ...

        def wrapped(
            self: TxrmBaseType | None,
            /,
            *args: typing.Any,
            strict: bool | None = None,
            **kwargs: typing.Any,
        ) -> RetType | None:
            strict = kwargs.get("strict")
            if strict is None:
                if _strict is None:
                    if self is None:
                        raise ValueError(
                            "Argument strict must be defined if self is None"
                        )
                    strict = self.strict
                else:
                    # Default set when calling uses_ole overrides class
                    strict = _strict

            try:
                return function(*args, **kwargs)
            except TxrmFileError:
                if strict:
                    raise TxrmFileError(
                        f"{function.__name__}: failed as file is not open."
                    )
                return None

        return wrapped

    if function is None:
        return wrapped_wrapper
    return wrapped_wrapper(function)
