import decimal
from abc import ABC, abstractmethod
import typing
from typing_extensions import Protocol, runtime_checkable

T = typing.TypeVar("T")

@runtime_checkable
class TypeCaster(Protocol[T]):
    """
    Protocol for a type casting method.

    A type caster must be able to be called with a single value, and return
    a single value, being the value coursed into to the correct type.

    Casting methods are also allowed to throw an exception when the value
    received can't be handled.
    """
    def __call__(self, param: typing.Any) -> T:
        pass

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self):
        return type(self).__name__


class DictConstructable(type):
    def __getitem__(cls, item):
        return cls.__call__(*item)


class ConvertibleTypeCaster(TypeCaster, ABC, metaclass=DictConstructable):
    @abstractmethod
    def __call__(self, arg: typing.Any) -> T:
        pass

    @abstractmethod
    def to_type_hint(self) -> typing.Type:
        pass


def to_hint(t: TypeCaster) -> typing.Type:
    if(isinstance(t, ConvertibleTypeCaster)):
        return t.to_type_hint()
    if(isinstance(t, type)):
        return t
    raise ValueError(f"Unable to convert '{t}' to a python type hint!")


class RangedInteger(ConvertibleTypeCaster):
    def __init__(self, minimum: float, maximum: float):
        self._min = float(minimum)
        self._max = float(maximum)

    def __call__(self, param: typing.Any) -> int:
        param = int(param)

        if(not (self._min <= param <= self._max)):
            raise ValueError(f"Value: '{param}' is not between {self._min} and {self._max}")

        return param

    def __repr__(self) -> str:
        return f"{type(self).__name__}[min={self._min}, max={self._max}]"

    def to_type_hint(self) -> typing.Type:
        return int


class RangedFloat(ConvertibleTypeCaster):
    def __init__(self, minimum: float, maximum: float):
        self._min = float(minimum)
        self._max = float(maximum)

    def __call__(self, param: typing.Any) -> float:
        param = float(param)

        if(not (self._min <= param <= self._max)):
            raise ValueError(f"Value: '{param}' is not between {self._min} and {self._max}")

        return param

    def __repr__(self) -> str:
        return f"{type(self).__name__}[min={self._min}, max={self._max}]"

    def to_type_hint(self) -> typing.Type:
        return float


class Sequence(ConvertibleTypeCaster):
    def __init__(self, item_type: typing.Callable[[typing.Any], typing.Any]):
        self._item_type = item_type

    def __call__(self, params: typing.Any) -> typing.Any:
        vals = []

        for param in params:
            try:
                vals.append(self._item_type(param))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Value: '{param}' can not be converted to {self._item_type}. Reason:\n{e}"
                )

        return vals

    def to_type_hint(self) -> typing.Type:
        return typing.List[to_hint(self._item_type)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self._item_type}]"


class Tuple(ConvertibleTypeCaster):
    def __init__(self, *type_list: typing.Callable[[typing.Any], typing.Any]):
        self._valid_type_list = type_list

    def __call__(self, params: typing.Any) -> typing.Any:
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

    def to_type_hint(self) -> typing.Type:
        return typing.Tuple[tuple(to_hint(t) for t in self._valid_type_list)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._valid_type_list}"


class Literal(ConvertibleTypeCaster):
    def __init__(self, *objects: typing.Any):
        self._valid_objs = objects

    def __call__(self, param: typing.Any) -> typing.Any:
        for obj in self._valid_objs:
            if(param == obj):
                return param

        raise ValueError(
            f"Value: '{param}' is not any one of the literals:"
            f"\n{self._valid_objs}"
        )

    def to_type_hint(self) -> typing.Type:
        return typing.Literal[tuple(self._valid_objs)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._valid_objs}"


NoneType = Literal[None]


class Union(ConvertibleTypeCaster):
    def __init__(self, *types: typing.Callable[[typing.Any], typing.Any]):
        self._valid_types = list(types)

    def __call__(self, param: typing.Any) -> typing.Any:
        for t in self._valid_types:
            try:
                return t(param)
            except (TypeError, ValueError) as e:
                continue

        raise ValueError(
            f"Value: '{param}' can not be converted to any of the following "
            f"types:\n{self._valid_types}"
        )

    def to_type_hint(self) -> typing.Type:
        return Union[tuple(to_hint(t) for t in self._valid_types)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._valid_types}"


class Optional(Union):
    def __init__(self, t: typing.Callable[[typing.Any], typing.Any]):
        super().__init__(t, NoneType)

    def to_type_hint(self) -> typing.Type:
        return typing.Optional[tuple(to_hint(t) for t in self._valid_types)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self._valid_types[0]}]"


class RoundedDecimal(ConvertibleTypeCaster):
    def __init__(self, precision: int = 5):
        self._precision = precision

    def __call__(self, param: typing.Any) -> float:
        return float(decimal.Decimal(
            param, context=decimal.Context(prec=self._precision)
        ))

    def to_type_hint(self) -> typing.Type:
        return float

    def __repr__(self) -> str:
        return f"{type(self).__name__}[precision={self._precision}]"


class Dict(ConvertibleTypeCaster):
    def __init__(self, key: typing.Callable[[typing.Any], typing.Any], value: typing.Callable[[typing.Any], typing.Any]):
        self._key = key
        self._value = value

    def __call__(self, param: typing.Any) -> dict:
        return {
            self._key(k): self._value(v) for k, v in dict(param).items()
        }

    def to_type_hint(self) -> typing.Type:
        return typing.Dict[to_hint(self._key), to_hint(self._value)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self._key}, {self._value}]"