import argparse
import sys
from argparse import ArgumentParser, Namespace
from typing import Callable, Literal, Type, Any, Union, Tuple, List, Dict, Optional
import inspect
import re
import yaml
from io import StringIO
from typing import get_origin, get_args
from diplomat.processing import ConfigSpec, TypeCaster


def _yaml_arg_load(str_list: List[str]) -> dict:
    if(not isinstance(str_list, list)):
        return str_list
    res = yaml.safe_load(StringIO(" ".join(str_list)))
    return res

def _yaml_dict(name: str, str_list: List[str]) -> dict:
    res = _yaml_arg_load(str_list)
    if(not isinstance(res, dict)):
        raise ValueError(f"Argument {name} specified is not a dict!")
    return res

def _yaml_list(name: str, str_list: List[str]):
    res = _yaml_arg_load(str_list)
    if(not isinstance(res, list)):
        raise ValueError(f"Argument {name} specified is not a list!")
    return res

def _yaml_union(union_type: Type):
    def checker(name: str, str_list: List[str]):
        res = _yaml_arg_load(str_list)
        valid_types = [t if(get_origin(t) is None) else get_origin(t) for t in get_args(union_type)]

        for t in valid_types:
            if(isinstance(res, t)):
                return res

        raise ValueError(f"Argument {name} is not of the type {union_type}")

    return checker

def _yaml_typecaster(caster: TypeCaster):
    def checker(name: str, str_list: List[str]):
        res = _yaml_arg_load(str_list)
        try:
            return caster(res)
        except Exception as e:
            raise type(e)(f"Failed to parse {name} because: {e}")

    return checker


def _func_arg_to_cmd_arg(annotation: Type, default: Any) -> Tuple[dict, Optional[Callable]]:
    arg_corrector = None

    if(Literal == get_origin(annotation)):
        # If the type is a literal, add choices...
        options = get_args(annotation)
        args = dict(choices=options, type=type(options[0]))
    elif(get_origin(annotation) == Union):
        args = dict(nargs="+", type=str)
        arg_corrector = _yaml_union(annotation)
    elif((get_origin(annotation) is not None and issubclass(get_origin(annotation), List)) or issubclass(annotation, List)):
        # List uses the odd yaml parsing...
        args = dict(nargs="+", type=str)
        arg_corrector = _yaml_list
    elif((get_origin(annotation) is not None and issubclass(get_origin(annotation), Dict)) or issubclass(annotation, Dict)):
        args = dict(nargs="+", type=str)
        arg_corrector = _yaml_dict
    elif(issubclass(annotation, bool)):
        args = dict(choices=(False, True), type=bool)
    elif(isinstance(annotation, TypeCaster)):
        args = dict(nargs="+", type=str)
        arg_corrector = _yaml_typecaster(annotation)
    else:
        if(not issubclass(annotation, Callable)):
            raise ValueError("Auto-CMD functions must have callable annotations if they are not of type list, dict, union, or literal.")
        args = dict(type=annotation)


    if(default == inspect.Parameter.empty):
        args["required"] = False
    else:
        args["default"] = default

    return args, arg_corrector


class ComplexParsingWrapper:
    def __init__(self, run_func: Callable, correctors: Dict[str, Callable]):
        self._func = run_func
        self._correctors = correctors

    def __call__(self, parsed_args: Namespace) -> Any:
        result = vars(parsed_args)

        for var, corrector in self._correctors.items():
            result[var] = corrector(var, result[var])

        return self._func(**result)


def func_to_command(func: Callable, parser: ArgumentParser) -> ArgumentParser:
    signature = inspect.signature(func)

    arg_correctors = {}

    # Extract params from the doc string...
    doc_str = inspect.getdoc(func)
    if(doc_str is None):
        help_messages = {}
    else:
        help_messages = {name: info for name, info in re.findall(":param +([a-zA-Z0-9_]+):([^:]*)", doc_str)}

    for name, param in signature.parameters.items():
        if(param.kind == inspect.Parameter.POSITIONAL_ONLY):
            raise ValueError("Function can only be turned into command if it has 0 positional only arguments!")
        if(param.kind in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]):
            continue

        if(param.annotation == inspect.Parameter.empty):
            raise ValueError("Auto-CLI function must annotate ALL variables!")

        args, corrector = _func_arg_to_cmd_arg(param.annotation, param.default)

        if(name in help_messages):
            args["help"] = help_messages[name]

        parser.add_argument("--" + name, **args)
        if(corrector is not None):
            arg_correctors[name] = corrector

    extra_args = getattr(func, "__extra_args", {})

    for name, (default, caster, desc) in extra_args.items():
        args, corrector = _func_arg_to_cmd_arg(caster, default)
        args["help"] = desc

        parser.add_argument("--" + name, **args)
        if(corrector is not None):
            arg_correctors[name] = corrector

    parser.set_defaults(_func=ComplexParsingWrapper(func, arg_correctors))

    return parser


class CLIEngine:
    def __init__(self, parent_parser: ArgumentParser):
        self._parser = parent_parser

    def __call__(self, arg_list: List[str]) -> Any:
        res = self._parser.parse_args(arg_list)
        func = getattr(res, "_func", None)
        if(func is not None):
            del res._func
            return func(res)
        else:
            self._parser.print_usage()


def build_full_parser(function_tree: dict, parent_parser: ArgumentParser) -> CLIEngine:
    sub_commands = parent_parser.add_subparsers()

    for command_name, sub_actions in function_tree.items():
        sub_parser = sub_commands.add_parser(command_name)

        if(isinstance(sub_actions, dict)):
            build_full_parser(sub_actions, sub_parser)
        else:
            func_to_command(sub_actions, sub_parser)

    return CLIEngine(parent_parser)


def extra_cli_args(config_spec: ConfigSpec) -> Callable[[Callable], Callable]:
    """
    A decorator for attaching additional CLI arguments to an auto-cli function...

    :param config_spec: The additional arguments to attach to the function, using config-spec format.

    :return: A decorator which attaches these arguments to the function, so they are included when turning it into a CLI function...
    """
    def attach_extra(func: Callable):
        func.__extra_args = config_spec
        return func
    return attach_extra



