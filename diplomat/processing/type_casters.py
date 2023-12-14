"""
Provides an interface and several types for performing automatic typecasting.
Also provides an alternative API for adding type hints to python functions
with more capabilities than python's builtin type hints via the typecaster
protocol and the typecaster_function decorator.
"""

import decimal
import os
import typing
from pathlib import Path
from typing_extensions import Protocol, runtime_checkable
import inspect


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
    """
    Protocol for representing a type caster function.
    """
    _type_casters: typing.Dict[str, TypeCaster]
    __type_caster_kwd_name: typing.Optional[str]

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
        """
        Construct this type caster using dictionary style construction.

        :param item: The arguments, can be a tuple.

        :return: A ConvertibleTypeCaster with the passed arguments passed to its constructor.
        """
        if(not isinstance(item, tuple)):
            item = (item,)
        return cls(*item)

    def __call__(self, arg: typing.Any) -> T:
        """
        Must be implemented by subclasses. Cast a value to this type.

        :param arg: The value to be casted.

        :return: The converted value, correctly converted to this type.

        :raises ValueError: If the passed value can't be properly converted to this type.
        """
        raise NotImplementedError()

    def to_metavar(self) -> str:
        """
        Convert this type to a CLI-Friendly format for display on the command line as a metavar.

        :return: A string, the CLI Friendly format for this type.
        """
        return repr(self).upper()

    def to_type_hint(self) -> typing.Type:
        """
        Abstract method: Convert this typecaster instance to a regular type hint.

        :return: A type from the typing module or primitive, being the underlying type
                 this typecaster converts values to and represents.
        """
        raise NotImplementedError()


class SingletonConvertibleTypeCaster(ConvertibleTypeCaster):
    def __init__(self):
        super().__init__()
        self.__doc__ = type(self).__doc__


def get_type_name(caster: TypeCaster) -> str:
    """
    Get the underlying name of a typecaster or type for printing...

    :param caster: The typecaster to get the string representation of.

    :return: A string, the representation of the type for display.
    """
    if(isinstance(caster, type)):
        return caster.__name__
    if(isinstance(caster, ConvertibleTypeCaster)):
        return repr(caster)
    return getattr(caster, "__name__", repr(caster))


def typecaster_function(func: typing.Callable) -> TypeCasterFunction:
    """
    Turns a function annotated with typecaster objects into a regular function
    with normal type annotations. The original typecaster annotations can be
    retrieved using :py:func:`~diplomat.processing.type_casters.get_typecaster_annotations`.

    :param func: The function to manipulate the typecaster based annotations of.

    :return: The original function with modified annotations and additional functionality for
             extracting the original type caster types...
    """
    if(hasattr(func, "__wrapped__")):
        raise TypeError("Can only typecaster annotate unwrapped functions, put this decorator first.")

    sig = inspect.signature(func)

    new_annotations = {}
    tc_config = {}
    wild_kwd_name = None

    # We assume all values are typecaster types...
    for name, param in sig.parameters.items():
        # We don't require type hints on **kwargs...
        if(param.kind == inspect.Parameter.POSITIONAL_ONLY):
            raise ValueError("Typecaster functions don't support positional only arguments!")
        if(param.kind == inspect.Parameter.VAR_POSITIONAL):
            raise ValueError("Typecaster functions don't support variable position arguments!")
        if(param.kind == inspect.Parameter.VAR_KEYWORD):
            wild_kwd_name = name
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
    func._type_caster_kwd_name = wild_kwd_name

    return func


def get_typecaster_annotations(func: TypeCasterFunction) -> typing.Dict[str, TypeCaster]:
    """
    Get the type casting annotations of a type caster function. This can be used for sanitizing command line arguments before passing them to this
    function.

    :param func: The type caster function to extract arguments from.

    :return: A dictionary of argument names to type caster types, describing the types of this type caster function.
    """
    res = getattr(func, "_type_casters", None)

    if(res is None):
        raise TypeError("Passed function was not a typecaster function!")

    return res


def get_typecaster_required_arguments(func: TypeCasterFunction) -> typing.Set[str]:
    """
    Get the arguments that must be passed to this typecasting function (one's that lack a default argument).

    :param func: The type caster function to extract required arguments from.

    :return: A set of strings, being the names of arguments that must be passed to this function for it to run.
    """
    ignore = [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]

    return {
        name for name, value in inspect.signature(func).parameters.items()
        if(value.default is inspect.Parameter.empty and value.kind not in ignore)
    }


def get_typecaster_kwd_arg_name(func: TypeCasterFunction) -> typing.Optional[str]:
    """
    Get the name of the wildcard keyword argument for this type caster function if it exists.

    :param func: The type caster function to extract the keyword argument name of.

    :returns: The name of the keyword argument, or None if this function has no wild keyword argument.
    """
    return getattr(func, "_type_caster_kwd_name", None)


def to_hint(t: TypeCaster) -> typing.Type:
    """
    Convert a type caster to a python type hint.

    :param t: The type caster hint to convert.

    :return: A type, repressing the underlying type this type caster represents.
    """
    if(isinstance(t, ConvertibleTypeCaster)):
        return t.to_type_hint()
    if(isinstance(t, type)):
        return t
    if(hasattr(t, "to_type_hint") and callable(t.to_type_hint)):
        return t.to_type_hint()

    raise ValueError(f"Unable to convert '{t}' to a python type hint!")


def to_metavar(t: TypeCaster) -> str:
    """
    Convert a type caster to a command line meta-variable string.

    :param t: The type caster.

    :return: A string representing the type on the command line.
    """
    if(isinstance(t, ConvertibleTypeCaster)):
        return t.to_metavar()
    elif(hasattr(t, "to_metavar") and callable(t.to_metavar)):
        return t.to_metavar()
    else:
        return get_type_name(t).upper()


class Any(SingletonConvertibleTypeCaster):
    """
    A type caster representing typing.Any. Passes all types through with no conversion.
    """
    def __call__(self, param: typing.Any) -> typing.Any:
        return param

    def __repr__(self) -> str:
        return f"{type(self).__name__}"

    def to_metavar(self) -> str:
        return "VAL"

    def to_type_hint(self) -> typing.Type:
        return typing.Any


Any: ConvertibleTypeCaster = Any()
"""A type caster representing typing.Any. Passes all types through with no conversion."""


class RangedInteger(ConvertibleTypeCaster):
    """
    Represents an integer with a restricted range of values it can take on.
    """
    def __init__(self, minimum: float, maximum: float):
        """
        Create a ranged integer type.

        :param minimum: The minimum allowed value of this integer, a float, inclusive.
        :param maximum: The maximum allowed value of this type, a float, inclusive.
        """
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

    def to_metavar(self) -> str:
        return "INT"

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

    def to_metavar(self) -> str:
        return "FLOAT"

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


class List(ConvertibleTypeCaster):
    """
    A type which represents a sequence, or list.
    """
    def __init__(self, item_type: typing.Callable[[typing.Any], typing.Any]):
        self._item_type = item_type

    def __call__(self, params: typing.Any) -> typing.Any:
        if(not isinstance(params, (list, tuple))):
            raise ValueError(f"Argument '{params}' is not a tuple or list!")

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
        if(isinstance(other, List)):
            return self._item_type == other._item_type
        return super().__eq__(other)

    def to_metavar(self) -> str:
        return f"[{to_metavar(self._item_type)}, ...]"

    def to_type_hint(self) -> typing.Type:
        return typing.List[to_hint(self._item_type)]

    def __hash__(self):
        return hash(self._item_type)

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{get_type_name(self._item_type)}]"


class Tuple(ConvertibleTypeCaster):
    """
    Represents a fixed length tuple of types.
    """
    def __init__(self, *type_list: TypeCaster):
        self._valid_type_list = type_list

    def __call__(self, params: typing.Any) -> typing.Any:
        vals = []

        if(len(params) != len(self._valid_type_list)):
            raise ValueError(f"Length of input is not "
                             f"{len(self._valid_type_list)}.")

        for param, v_type in zip(params, self._valid_type_list):
            try:
                 vals.append(v_type(param))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Value: '{param}' is not of type:\n{v_type}"
                )

        return tuple(vals)

    def to_type_hint(self) -> typing.Type:
        return typing.Tuple[tuple(to_hint(t) for t in self._valid_type_list)]

    def to_metavar(self) -> str:
        return "[" + ", ".join(to_metavar(t) for t in self._valid_type_list) + "]"

    def __eq__(self, other):
        if(isinstance(other, Tuple)):
            return self._valid_type_list == other._valid_type_list
        return super().__eq__(other)

    def __hash__(self):
        return hash(self._valid_type_list)

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{', '.join([get_type_name(t) for t in self._valid_type_list])}]"


class Literal(ConvertibleTypeCaster):
    """
    Represents the typing.Literal type as a type caster.
    """
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
        try:
            return typing.Literal[tuple(self._valid_objs)]
        except AttributeError:
            import typing_extensions
            return typing_extensions.Literal[tuple(self._valid_objs)]

    def to_metavar(self) -> str:
        return "|".join(repr(t) for t in self._valid_objs)

    def __eq__(self, other):
        if(isinstance(other, Literal)):
            return frozenset(self._valid_objs) == frozenset(other._valid_objs)
        return super().__eq__(other)

    def __hash__(self):
        return hash(frozenset(self._valid_objs))

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._valid_objs}"


class NoneType(SingletonConvertibleTypeCaster):
    """
    Represents None as a type caster.
    """
    def __call__(self, param: typing.Any) -> None:
        if(param is not None):
            raise ValueError("Value passed was not None!")
        return param

    def to_type_hint(self) -> None:
        return None

    def __repr__(self) -> str:
        return "None"


NoneType: ConvertibleTypeCaster = NoneType()
"""Represents None as a type caster."""


class Union(ConvertibleTypeCaster):
    """
    Represents the typing.Union type as a type caster.
    """
    def __init__(self, *types: TypeCaster):
        self._valid_types = types

    def __call__(self, param: typing.Any) -> typing.Any:
        for t in self._valid_types:
            try:
                return t(param)
            except (TypeError, ValueError):
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

    def to_metavar(self) -> str:
        return "|".join({to_metavar(v): 0 for v in self._valid_types}.keys())

    def to_type_hint(self) -> typing.Type:
        return typing.Union[tuple(to_hint(t) for t in self._valid_types)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{', '.join([get_type_name(t) for t in self._valid_types])}]"


class Optional(Union):
    """
    Represents typing.Optional as a type caster.
    """
    def __init__(self, t: TypeCaster):
        super().__init__(NoneType, t)

    def to_metavar(self) -> str:
        return to_metavar(self._valid_types[1])

    def to_type_hint(self) -> typing.Type:
        return typing.Optional[to_hint(self._valid_types[1])]

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{get_type_name(self._valid_types[1])}]"


class RoundedDecimal(ConvertibleTypeCaster):
    """
    Represents a decimal rounded to a fixed precision.
    """
    def __init__(self, precision: int = 5):
        self._precision = precision

    def __call__(self, param: typing.Any) -> float:
        return float(decimal.Decimal(
            param, context=decimal.Context(prec=self._precision)
        ))

    def to_type_hint(self) -> typing.Type:
        return float

    def to_metavar(self) -> str:
        return "FLOAT"

    def __eq__(self, other):
        if(isinstance(other, RoundedDecimal)):
            return self._precision == other._precision
        return super().__eq__(other)

    def __hash__(self):
        return hash(self._precision)

    def __repr__(self) -> str:
        return f"{type(self).__name__}[precision={self._precision}]"


class Dict(ConvertibleTypeCaster):
    """
    Represents typing.Dict as a type caster
    """
    def __init__(self, key: TypeCaster, value: TypeCaster):
        self._key = key
        self._value = value

    def __call__(self, param: typing.Any) -> dict:
        return {
            self._key(k): self._value(v) for k, v in dict(param).items()
        }

    def to_type_hint(self) -> typing.Type:
        return typing.Dict[to_hint(self._key), to_hint(self._value)]

    def to_metavar(self) -> str:
        return f"{{{to_metavar(self._key)}: {to_metavar(self._value)}, ...}}"

    def __eq__(self, other):
        if(isinstance(other, Dict)):
            return self._key == other._key and self._value == other._value
        return super().__eq__(other)

    def __hash__(self):
        return hash((self._key, self._value))

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{get_type_name(self._key)}, {get_type_name(self._value)}]"


class StrictCallable(ConvertibleTypeCaster):
    """
    A type caster that can be used to run strict argument name and type checking on type casting functions. Useful for API conformance checks.
    """
    def __init__(self, *, _return: TypeCaster = NoneType, _kwargs: bool = False, **kwargs: TypeCaster):
        self._return_type = _return
        self._required_args = kwargs
        self._wild_kwargs_req = _kwargs

    def __call__(self, arg: typing.Any) -> typing.Callable:
        if(not callable(arg)):
            raise TypeError("Passed argument a callable!")
        # Check for the argument values....
        annots = get_typecaster_annotations(arg)

        if(self._wild_kwargs_req):
            if(get_typecaster_kwd_arg_name(arg) is None):
                raise ValueError(
                    "Passed callable does not specify a variable keyword argument "
                    "(**kwargs), which is required by this callable."
                )

        for name, expected_annot in self._required_args.items():
            if(name not in annots):
                raise TypeError(f"Callable does not have an argument called: {name}")
            if(not (expected_annot == annots[name])):
                raise TypeError(f"Argument '{name}' annotation '{annots[name]}' does not match '{expected_annot}'")

        if(not (self._return_type == annots["return"])):
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
        return f"{type(self).__name__}({', '.join(k + ': ' + get_type_name(v) for k, v in self._required_args.items())})"


class PathLike(Union, SingletonConvertibleTypeCaster):
    """
    Represents os.PathLike as a type caster.
    """
    def __init__(self):
        super().__init__(Path, str)

    def __call__(self, arg: typing.Any) -> Path:
        return Path(arg)

    def __repr__(self):
        return type(self).__name__

    def to_metavar(self) -> str:
        return "FILE"

    def to_type_hint(self) -> typing.Type:
        return Union[os.PathLike, str]


PathLike: ConvertibleTypeCaster = PathLike()
"""Represents os.PathLike as a type caster."""
