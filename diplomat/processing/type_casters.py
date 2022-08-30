"""
Provides an interface and several types for performing automatic typecasting.
Also provides an alternative API for adding type hints to python functions
with more capabilities than python's builtin type hints via the typecaster
protocol and the typecaster_function decorator.
"""

import decimal
import typing
from typing_extensions import Protocol, runtime_checkable
import inspect
from pathlib import Path


T = typing.TypeVar("T")

@runtime_checkable
class TypeCaster(Protocol[T]):
    """
    Protocol for a type casting method. These are 'smart' types, capable
    of checking, and converting other types into their own type via the
    call operator.

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


class TypeCasterFunction(Protocol):
    _type_casters: typing.Dict[str, TypeCaster]

    def __call__(self, *args, **kwargs) -> typing.Any:
        pass


class ConvertibleTypeCaster(TypeCaster):
    """
    A convertible typecaster, or one that can be converted to normal python
    type hints. This class also enables square bracket construction.
    (TypeCaster[args] is the same as TypeCaster(args)) This class is used
    for handling non-trivial type hinting types (such as Union, Dict, List, etc.)
    """
    def __class_getitem__(cls, item):
        if(not isinstance(item, tuple)):
            item = (item,)
        return cls(*item)

    def __call__(self, arg: typing.Any) -> T:
        raise NotImplementedError()

    def to_type_hint(self) -> typing.Type:
        """
        Abstract method: Convert this typecaster instance to a regular type hint.

        :return: A type from the typing module or primitive, being the underlying type
                 this typecaster converts values to and represents.
        """
        raise NotImplementedError()


def typecaster_function(func: typing.Callable) -> TypeCasterFunction:
    """
    Turns a function annotated with typecaster objects into a regular function
    with normal type annotations. The original typecaster annotations can be
    retrieved using the get_typecaster_annotations in this same module.

    :param func: The function to manipulate the typecaster based annotations of.

    :return: The original function with modified annotations and additional functionality for
             extracting the original type casters...
    """
    if(hasattr(func, "__wrapped__")):
        raise TypeError("Can only typecaster annotate unwrapped functions, put this decorator first.")

    sig = inspect.signature(func)

    new_annotations = {}
    tc_config = {}

    # We assume all values are typecaster types...
    for name, param in sig.parameters.items():
        # We don't require type hints on **kwargs...
        if(param.kind == inspect.Parameter.POSITIONAL_ONLY):
            raise ValueError("Typecaster functions don't support positional only arguments!")
        if(param.kind == inspect.Parameter.VAR_POSITIONAL or param.kind == inspect.Parameter.VAR_KEYWORD):
            continue
        if(param.annotation == inspect.Parameter.empty):
            raise ValueError("Typecaster annotated functions must annotate all input arguments!")

        new_annotations[name] = to_hint(param.annotation)
        tc_config[name] = param.annotation

    ret_ann = NoneType if(sig.return_annotation == inspect.Signature.empty) else sig.return_annotation
    new_annotations["return"] = to_hint(ret_ann)
    tc_config["return"] = ret_ann

    # Now we override the default annotations, and move the original annotations to a new attribute.
    # This can be used by other code
    func.__annotations__ = new_annotations
    func._type_casters = tc_config

    return func


def get_typecaster_annotations(func: TypeCasterFunction) -> typing.Dict[str, TypeCaster]:
    res = getattr(func, "_type_casters", None)

    if(res is None):
        raise TypeError("Passed function was not a typecaster function!")

    return res


def to_hint(t: TypeCaster) -> typing.Type:
    if(isinstance(t, ConvertibleTypeCaster)):
        return t.to_type_hint()
    if(isinstance(t, type)):
        return t
    raise ValueError(f"Unable to convert '{t}' to a python type hint!")


class Any(ConvertibleTypeCaster):
    def __call__(self, param: typing.Any) -> typing.Any:
        return param

    def __repr__(self) -> str:
        return f"{type(self).__name__}"

    def to_type_hint(self) -> typing.Type:
        return typing.Any

Any = Any()

class RangedInteger(ConvertibleTypeCaster):
    def __init__(self, minimum: float, maximum: float):
        self._min = float(minimum)
        self._max = float(maximum)

    def __call__(self, param: typing.Any) -> int:
        param = int(param)

        if(not (self._min <= param <= self._max)):
            raise ValueError(f"Value: '{param}' is not between {self._min} and {self._max}")

        return param

    def __eq__(self, other):
        if(isinstance(other, RangedInteger)):
            return self._min == other._min and self._max == other._max
        return super().__eq__(other)

    def __hash__(self):
        return hash((self._min, self._max))

    def __repr__(self) -> str:
        return f"{type(self).__name__}[min={self._min}, max={self._max}]"

    def to_type_hint(self) -> typing.Type:
        return int


class RangedFloat(ConvertibleTypeCaster):
    """
    Represents a float with a restricted range of values it can take on.
    """
    def __init__(self, minimum: float, maximum: float):
        """
        Create a ranged float, allowing values between minimum and maximum.

        :param minimum: The minimum value allowed for the float, inclusive.
        :param maximum: The maximum value allowed for the float, inclusive.
        """
        self._min = float(minimum)
        self._max = float(maximum)

    def __call__(self, param: typing.Any) -> float:
        param = float(param)

        if(not (self._min <= param <= self._max)):
            raise ValueError(f"Value: '{param}' is not between {self._min} and {self._max}")

        return param

    def __eq__(self, other):
        if(isinstance(other, RangedFloat)):
            return self._min == other._min and self._max == other._max
        return super().__eq__(other)

    def __hash__(self):
        return hash((self._min, self._max))

    def __repr__(self) -> str:
        return f"{type(self).__name__}[min={self._min}, max={self._max}]"

    def to_type_hint(self) -> typing.Type:
        return float


class Sequence(ConvertibleTypeCaster):
    """
    A type which represents a sequence, or list.
    """
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

    def __eq__(self, other):
        if(isinstance(other, Sequence)):
            return self._item_type == other._item_type
        return super().__eq__(other)

    def to_type_hint(self) -> typing.Type:
        return typing.List[to_hint(self._item_type)]

    def __hash__(self):
        return hash(self._item_type)

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self._item_type}]"


class Tuple(ConvertibleTypeCaster):
    def __init__(self, *type_list: TypeCaster):
        self._valid_type_list = list(type_list)

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

    def __eq__(self, other):
        if(isinstance(other, Tuple)):
            return self._valid_type_list == other._valid_type_list
        return super().__eq__(other)

    def __hash__(self):
        return hash(self._valid_type_list)

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._valid_type_list}"


class Literal(ConvertibleTypeCaster):
    def __init__(self, *objects: typing.Any):
        self._valid_objs = list(objects)

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

    def __eq__(self, other):
        if(isinstance(other, Literal)):
            return frozenset(self._valid_objs) == frozenset(other._valid_objs)
        return super().__eq__(other)

    def __hash__(self):
        return hash(frozenset(self._valid_objs))

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._valid_objs}"


class NoneType(ConvertibleTypeCaster):
    def __call__(self, param: typing.Any) -> None:
        if(param is not None):
            raise ValueError("Value passed was not None!")
        return param

    def to_type_hint(self) -> None:
        return None

    def __repr__(self) -> str:
        return "None"

NoneType = NoneType()


class Union(ConvertibleTypeCaster):
    def __init__(self, *types: TypeCaster):
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

    def __eq__(self, other):
        if(isinstance(other, Union)):
            return frozenset(self._valid_types) == frozenset(other._valid_types)
        return super().__eq__(other)

    def __hash__(self):
        return hash(frozenset(self._valid_types))

    def to_type_hint(self) -> typing.Type:
        return typing.Union[tuple(to_hint(t) for t in self._valid_types)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._valid_types}"


class Optional(Union):
    def __init__(self, t: TypeCaster):
        super().__init__(NoneType, t)

    def to_type_hint(self) -> typing.Type:
        return typing.Optional[to_hint(self._valid_types[1])]

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self._valid_types[1]}]"


class RoundedDecimal(ConvertibleTypeCaster):
    def __init__(self, precision: int = 5):
        self._precision = precision

    def __call__(self, param: typing.Any) -> float:
        return float(decimal.Decimal(
            param, context=decimal.Context(prec=self._precision)
        ))

    def to_type_hint(self) -> typing.Type:
        return float

    def __eq__(self, other):
        if(isinstance(other, RoundedDecimal)):
            return self._precision == other._precision
        return super().__eq__(other)

    def __hash__(self):
        return hash(self._precision)

    def __repr__(self) -> str:
        return f"{type(self).__name__}[precision={self._precision}]"


class Dict(ConvertibleTypeCaster):
    def __init__(self, key: TypeCaster, value: TypeCaster):
        self._key = key
        self._value = value

    def __call__(self, param: typing.Any) -> dict:
        return {
            self._key(k): self._value(v) for k, v in dict(param).items()
        }

    def to_type_hint(self) -> typing.Type:
        return typing.Dict[to_hint(self._key), to_hint(self._value)]

    def __eq__(self, other):
        if(isinstance(other, Dict)):
            return self._key == other._key and self._value == other._value
        return super().__eq__(other)

    def __hash__(self):
        return hash((self._key, self._value))

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self._key}, {self._value}]"


class StrictCallable(ConvertibleTypeCaster):
    def __init__(self, *, _return: TypeCaster = NoneType, **kwargs: TypeCaster):
        self._return_type = _return
        self._required_args = kwargs

    def __call__(self, arg: typing.Any) -> typing.Callable:
        if(not callable(arg)):
            raise TypeError("Passed argument a callable!")
        # Check for the argument values....
        annots = get_typecaster_annotations(arg)

        for name, expected_annot in self._required_args.items():
            if(name not in annots):
                raise TypeError(f"Callable does not have an argument called: {name}")
            if(not (annots[name] == expected_annot)):
                raise TypeError(f"Argument '{name}' annotation '{annots[name]}' does not match '{expected_annot}'")

        if(not (annots["return"] == self._return_type)):
            raise TypeError(f"Return annotation '{annots['return']}' does not match '{self._return_type}'")

        return arg

    def __eq__(self, other):
        if(isinstance(other, StrictCallable)):
            return self._required_args == other._required_args and self._return_type == other._return_type
        return super().__eq__(other)

    def __hash__(self):
        return hash((self._required_args, self._return_type))

    def to_type_hint(self) -> typing.Type:
        return typing.Callable

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(k + ': ' + repr(v) for k, v in self._required_args.items())})"


PathLike = Union[Path, str]