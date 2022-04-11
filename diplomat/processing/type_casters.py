import decimal
from typing import Any, Callable, TypeVar
from typing_extensions import Protocol

T = TypeVar("T")


class TypeCaster(Protocol[T]):
    """
    Protocol for a type casting method.

    A type caster must be able to be called with a single value, and return
    a single value, being the value coarsed into to the correct type.

    Casting methods are also allowed to throw an exception when the value
    received can't be handled.
    """
    def __call__(self, param: Any) -> T:
        pass

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self):
        return type(self).__name__


class RangedInteger(TypeCaster):
    def __init__(self, minimum: float, maximum: float):
        self._min = float(minimum)
        self._max = float(maximum)

    def __call__(self, param: Any) -> int:
        param = int(param)

        if(not (self._min <= param <= self._max)):
            raise ValueError(f"Value: '{param}' is not between {self._min} and {self._max}")

        return param

    def __repr__(self) -> str:
        return f"{type(self).__name__}[min={self._min}, max={self._max}]"


class RangedFloat(TypeCaster):
    def __init__(self, minimum: float, maximum: float):
        self._min = float(minimum)
        self._max = float(maximum)

    def __call__(self, param: Any) -> float:
        param = float(param)

        if(not (self._min <= param <= self._max)):
            raise ValueError(f"Value: '{param}' is not between {self._min} and {self._max}")

        return param

    def __repr__(self) -> str:
        return f"{type(self).__name__}[min={self._min}, max={self._max}]"


class Sequence(TypeCaster):
    def __init__(self, item_type: Callable[[Any], Any]):
        self._item_type = item_type

    def __call__(self, params: Any) -> Any:
        vals = []

        for param in params:
            try:
                vals.append(self._item_type(param))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Value: '{param}' can not be converted to {self._item_type}. Reason:\n{e}"
                )

        return vals

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self._item_type}]"


class Tuple(TypeCaster):
    def __init__(self, *type_list: Callable[[Any], Any]):
        self._valid_type_list = type_list

    def __call__(self, params: Any) -> Any:
        vals = []

        if(len(params) != len(self._valid_type_list)):
            raise ValueError(f"Length of input is not "
                             f"{len(self._valid_type_list)}.")

        for param, v_type in zip(params, self._valid_type_list):
            try:
                 vals.append(v_type(param))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Value: '{param}' is not of type:\n{v_type}"
                )

        return tuple(vals)

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._valid_type_list}"


class Literal(TypeCaster):
    def __init__(self, *objects: Any):
        self._valid_objs = objects

    def __call__(self, param: Any) -> Any:
        for obj in self._valid_objs:
            if(param == obj):
                return param

        raise ValueError(
            f"Value: '{param}' is not any one of the literals:"
            f"\n{self._valid_objs}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._valid_objs}"


class Union(TypeCaster):
    def __init__(self, *types: Callable[[Any], Any]):
        self._valid_types = types

    def __call__(self, param: Any) -> Any:
        for t in self._valid_types:
            try:
                return t(param)
            except (TypeError, ValueError) as e:
                continue

        raise ValueError(
            f"Value: '{param}' can not be converted to any of the following "
            f"types:\n{self._valid_types}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._valid_types}"


class RoundedDecimal(TypeCaster):
    def __init__(self, precision: int = 5):
        self._precision = precision

    def __call__(self, param: Any) -> decimal.Decimal:
        return decimal.Decimal(
            param, context=decimal.Context(prec=self._precision)
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}[precision={self._precision}]"