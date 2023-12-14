"""
Provides functions for turning typecaster annotated functions into CLI commands.
"""

from argparse import ArgumentParser, Namespace, HelpFormatter, Action, ONE_OR_MORE
from typing import Callable, Any, Tuple, List, Dict, Optional, Type
import inspect
import re
import yaml
from io import StringIO
from diplomat.processing import ConfigSpec
from diplomat.processing.type_casters import (
    TypeCaster,
    TypeCasterFunction,
    get_typecaster_annotations,
    get_type_name,
    get_typecaster_kwd_arg_name,
    to_metavar,
    ConvertibleTypeCaster
)


class CLIError(Exception):
    """
    A custom exception thrown when an error occurs when attempting to parse user CLI inputs. Used for handling cli
    parsing error gracefully internally.
    """
    pass


class Flag(ConvertibleTypeCaster):
    """
    Custom type caster type that represents a boolean flag argument on the command line (true/false doesn't need to be
    specified). It's python type is automatically converted to a boolean.
    """
    def __call__(self, arg: Any) -> bool:
        return bool(arg)

    def to_type_hint(self) -> Type:
        return bool

    def __repr__(self):
        return type(self).__name__


Flag = Flag()


class YAMLArgHelpFormatter(HelpFormatter):
    def _format_args(self, action: Action, default_metavar: str) -> str:
        get_metavar = self._metavar_formatter(action, default_metavar)

        if(action.nargs == ONE_OR_MORE):
            return f"{get_metavar(1)[0]}"

        super()._format_args(action, default_metavar)


def _yaml_arg_load(str_list: List[str]) -> dict:
    if(not isinstance(str_list, list)):
        return str_list

    str_list = " ".join(str_list)
    try:
        res = yaml.safe_load(StringIO(str_list))
    except Exception as e:
        raise CLIError(f"Unable to parse argument '{str_list}' as YAML, because: '{e}'")

    return res


def _yaml_typecaster(caster: TypeCaster):
    def checker(name: str, str_list: List[str]):
        res = _yaml_arg_load(str_list)
        try:
            return caster(res)
        except Exception as e:
            raise CLIError(f"Failed to parse {name}, because: '{e}'")

    return checker


def _func_arg_to_cmd_arg(annotation: TypeCaster, default: Any, auto_cast: bool = True) -> Tuple[dict, Optional[Callable]]:
    if(annotation is Flag):
        args = dict(action="store_true")
        arg_corrector = None
    else:
        args = dict(nargs="+", type=str, metavar=to_metavar(annotation))
        arg_corrector = _yaml_typecaster(annotation) if(auto_cast) else _yaml_typecaster(lambda a: a)

    if(default == inspect.Parameter.empty):
        args["required"] = True
    else:
        args["default"] = default

    return args, arg_corrector


class ComplexParsingWrapper:
    DELETE = object()

    def __init__(self, run_func: Callable, correctors: Dict[str, Callable], parser: ArgumentParser):
        self._func = run_func
        self._correctors = correctors
        self._parser = parser

    @property
    def parser(self) -> ArgumentParser:
        return self._parser

    @property
    def accepts_extra_flags(self) -> bool:
        return getattr(self._func, "__allow_arbitrary_flags", False)

    @property
    def correctors(self) -> Dict[str, Callable]:
        return self._correctors

    def __call__(self, parsed_args: Namespace) -> Any:
        result = vars(parsed_args)

        for var, value in list(result.items()):
            if(value is self.DELETE):
                del result[var]
                del self._correctors[var]

        for var, corrector in self._correctors.items():
            result[var] = corrector(var, result[var])

        return self._func(**result)


def get_summary_from_doc_str(doc_str: str) -> str:
    return "".join(re.split(":param |:return|:throw", doc_str)[:1])


def func_to_command(func: TypeCasterFunction, parser: ArgumentParser) -> ArgumentParser:
    parser.formatter_class = YAMLArgHelpFormatter
    parser.allow_abbrev = False
    signature = inspect.signature(func)
    cmd_args = get_typecaster_annotations(func)

    arg_correctors = {}

    # Extract params from the doc string...
    doc_str = inspect.getdoc(func)

    if(doc_str is None):
        help_messages = {}
    else:
        parser.description = get_summary_from_doc_str(doc_str)
        help_messages = {name: info for name, info in re.findall(":param +([a-zA-Z0-9_]+):([^:]*)", doc_str)}

    abbr_set = set()

    if(getattr(func, "__allow_arbitrary_flags", False)):
        name = get_typecaster_kwd_arg_name(func)
        if(name is not None and name in help_messages):
            parser.epilog = help_messages[name]

    pos_arg_count = getattr(func, "__pos_cmd_arg_count", 0)

    for name, caster in cmd_args.items():
        if(name == "return"):
            continue

        args, corrector = _func_arg_to_cmd_arg(caster, signature.parameters[name].default)

        if(name in help_messages):
            args["help"] = help_messages[name]

        abbr_cmd = "-" + "".join(s[:1] for s in name.split("_"))
        if(pos_arg_count > 0):
            if("nargs" in args):
                if(pos_arg_count > 1):
                    args["nargs"] = 1
                else:
                    # A default argument for positional arguments only works if the argument is in the last position.
                    no_default = signature.parameters[name].default is inspect.Parameter.empty
                    args["nargs"] = "+" if(no_default) else "*"
            parser.add_argument(name, **args)
            pos_arg_count -= 1
        elif(abbr_cmd in abbr_set):
            parser.add_argument("--" + name, **args)
        else:
            parser.add_argument("--" + name, abbr_cmd, **args)
            abbr_set.add(abbr_cmd)

        if(corrector is not None):
            arg_correctors[name] = corrector

    extra_args = getattr(func, "__extra_args", {})
    auto_cast = getattr(func, "__auto_cast", True)

    for name, (default, caster, desc) in extra_args.items():
        args, corrector = _func_arg_to_cmd_arg(caster, ComplexParsingWrapper.DELETE, auto_cast=auto_cast)
        args["help"] = desc

        parser.add_argument("--" + name, **args)
        if(corrector is not None):
            arg_correctors[name] = corrector

    parser.set_defaults(_func=ComplexParsingWrapper(func, arg_correctors, parser))

    return parser


class CLIEngine:
    def __init__(self, parent_parser: ArgumentParser):
        self._parser = parent_parser

    def _reparse(self, args: List[str], extra: List[str], arg_handler: ComplexParsingWrapper) -> Namespace:
        if(not arg_handler.accepts_extra_flags):
            return self._parser.parse_args(args)

        for op in extra:
            if(op.startswith("--")):
                name = op.split('=')[0]
                if(len(name) <= 2):
                    continue
                arg_handler.parser.add_argument(name, type=str, nargs="+", metavar="Unknown")
                arg_handler.correctors[name[2:]] = _yaml_typecaster(lambda a: a)

        return self._parser.parse_args(args)

    def __call__(self, arg_list: List[str]) -> Any:
        try:
            res, extra = self._parser.parse_known_args(arg_list)
        except TypeError as e:
            # Python 3.7 argparse doesn't handle subcommand namespaces correctly when no arguments are passed to them
            # (throws type error), we insert an empty string argument and reparse to get a more helpful error message
            # and force argparse to print the usage string...
            if(not (str(e) == "sequence item 0: expected str instance, NoneType found")):
                raise
            res, extra = self._parser.parse_known_args([*arg_list, ""])
        func = getattr(res, "_func", None)

        if(func is not None):
            if(extra):
                # Attempt to reparse after adding the extra arguments in
                # (if this is a function that accepts arbitrary flags)...
                res = self._reparse(arg_list, extra, func)
            del res._func
            try:
                return func(res)
            except CLIError as e:
                print(e)
                self._parser.print_usage()
        else:
            self._parser.print_usage()


def build_full_parser(function_tree: dict, parent_parser: ArgumentParser, name: Optional[str] = None) -> CLIEngine:
    name = parent_parser.prog if(name is None) else name
    parent_parser.allow_abbrev = False
    sub_commands = parent_parser.add_subparsers(title=f"Subcommands and namespaces of '{name}'", required=True)

    for command_name, sub_actions in function_tree.items():
        if(command_name.startswith("_")):
            continue

        if(isinstance(sub_actions, dict)):
            sub_cmd_args = {key[2:]: value for key, value in sub_actions.items() if (key.startswith("__"))}
            if("description" in sub_cmd_args):
                sub_cmd_args["help"] = sub_cmd_args["description"]

            sub_parser = sub_commands.add_parser(command_name, **sub_cmd_args)
            build_full_parser(sub_actions, sub_parser, name + " " + command_name)
        else:
            doc_str = inspect.getdoc(sub_actions)
            if(doc_str is not None):
                desc = get_summary_from_doc_str(doc_str)
                sub_parser = sub_commands.add_parser(command_name, description=desc, help=desc)
            else:
                sub_parser = sub_commands.add_parser(command_name)

            func_to_command(sub_actions, sub_parser)

    return CLIEngine(parent_parser)


def extra_cli_args(config_spec: ConfigSpec, auto_cast: bool = True) -> Callable[[Callable], Callable]:
    """
    A decorator for attaching additional CLI arguments to an auto-cli function...

    :param config_spec: The additional arguments to attach to the function, using config-spec format.
    :param auto_cast: Boolean flag, if true don't automatically convert extra cli args to their correct types.
                      This means the method needs to do the conversion itself.

    :return: A decorator which attaches these arguments to the function, so they are included when turning it into a
             CLI function...
    """
    def attach_extra(func: Callable):
        func.__extra_args = config_spec
        func.__auto_cast = auto_cast

        if(hasattr(func, "__doc__") and (func.__doc__ is not None)):
            doc_str_lst = func.__doc__.split("\n")
            magic_str = "{extra_cli_args}"

            for i, line in enumerate(doc_str_lst):
                escaped_line = line.replace("{{", "?").replace("}}", "?")
                index = escaped_line.find(magic_str)
                if(index == -1):
                    continue

                extra_doc = ("\n" + (" " * index)).join(
                    f" - {name} (Type: {get_type_name(caster)}, Default: {default}): {desc}"
                    for name, (default, caster, desc) in config_spec.items()
                )
                doc_str_lst[i] = line.format(extra_cli_args=extra_doc)

            func.__doc__ = "\n".join(doc_str_lst)

        return func

    return attach_extra


def allow_arbitrary_flags(func: Callable) -> Callable:
    func.__allow_arbitrary_flags = True
    return func


def positional_argument_count(amt: int) -> Callable[[Callable], Callable]:
    def attach_pos_args(func: Callable) -> Callable:
        func.__pos_cmd_arg_count = amt
        return func
    return attach_pos_args

