from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from .mixins.file import FileMixin

if TYPE_CHECKING:
    from typing import Generic, Callable, Any, Never, TypeVar, cast, overload

    ClassVar = TypeVar("ClassVar", bound=FileMixin)
    T = TypeVar("T", Any, None)
    RetType = TypeVar("RetType")


class TxrmProperty(Generic[RetType]):

    def __init__(self, fn: Callable[[ClassVar], RetType], fallback: T) -> None:
        self.fn = fn
        self.fallback = fallback
        self.hidden_var_str = f"_{self.fn.__name__}"

    def __get__(self, obj: ClassVar, object_type: type | None = None) -> RetType | T:
        try:
            if not self.hidden_var_str in obj.__txrm_properties:
                obj.__txrm_properties[self.hidden_var_str] = self.fn(obj)
        except Exception:
            if obj.strict:
                raise
            logging.error(
                "Failed to get property '%s'",
                self.fn.__name__,
                exc_info=True,
            )
        return cast(
            RetType | T, obj.__txrm_properties.get(self.hidden_var_str, self.fallback)
        )

    def __set__(self, obj: ClassVar, value: Any) -> Never:
        raise AttributeError(f"{self.fn.__name__} cannot be set.")

    def __delete__(self, obj: ClassVar) -> None:
        setattr(obj, self.hidden_var_str, None)


@overload
def txrm_property(
    function: None = None,
    fallback: T = None,
) -> Callable[[Callable[[ClassVar], RetType]], TxrmProperty[RetType]]: ...


@overload
def txrm_property(
    function: Callable[[ClassVar], RetType],
    fallback: T = None,
) -> TxrmProperty[RetType]: ...


def txrm_property(
    function: Callable[[ClassVar], RetType] | None = None,
    fallback: T | None = None,
) -> (
    TxrmProperty[RetType]
    | Callable[[Callable[[ClassVar], RetType]], TxrmProperty[RetType]]
):

    if function is None:

        def wrapper(function: Callable[[ClassVar], RetType]) -> TxrmProperty[RetType]:
            return TxrmProperty(function, fallback)

        return wrapper

    return TxrmProperty(function, fallback)
