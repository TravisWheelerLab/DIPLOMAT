from argparse import ArgumentParser, Namespace
from typing import Callable, Any, Tuple, List, Dict, Optional
import inspect
import re
import yaml
from io import StringIO
from diplomat.processing import ConfigSpec
from diplomat.processing.type_casters import TypeCaster, TypeCasterFunction, get_typecaster_annotations


def _yaml_arg_load(str_list: List[str]) -> dict:
    if(not isinstance(str_list, list)):
        return str_list
    res = yaml.safe_load(StringIO(" ".join(str_list)))
    return res


def _yaml_typecaster(caster: TypeCaster):
    def checker(name: str, str_list: List[str]):
        res = _yaml_arg_load(str_list)
        try:
            return caster(res)
        except Exception as e:
            raise type(e)(f"Failed to parse {name} because: {e}")

    return checker


def _func_arg_to_cmd_arg(annotation: TypeCaster, default: Any, auto_cast: bool = True) -> Tuple[dict, Optional[Callable]]:
    args = dict(nargs="+", type=str)
    arg_corrector = _yaml_typecaster(annotation) if(auto_cast) else _yaml_typecaster(lambda a: a)

    if(default == inspect.Parameter.empty):
        args["required"] = False
    else:
        args["default"] = default

    return args, arg_corrector


class ComplexParsingWrapper:
    DELETE = object()

    def __init__(self, run_func: Callable, correctors: Dict[str, Callable]):
        self._func = run_func
        self._correctors = correctors

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

    for name, caster in cmd_args.items():
        if(name == "return"):
            continue

        args, corrector = _func_arg_to_cmd_arg(caster, signature.parameters[name].default)

        if(name in help_messages):
            args["help"] = help_messages[name]

        abbr_cmd = "-" + "".join(s[:1] for s in name.split("_"))
        if(abbr_cmd in abbr_set):
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


def build_full_parser(function_tree: dict, parent_parser: ArgumentParser, name: Optional[str] = None) -> CLIEngine:
    name = parent_parser.prog if(name is None) else name
    sub_commands = parent_parser.add_subparsers(title=f"Subcommands and namespaces of '{name}'", required=True)

    for command_name, sub_actions in function_tree.items():
        if(command_name.startswith("__")):
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

    :return: A decorator which attaches these arguments to the function, so they are included when turning it into a CLI function...
    """
    def attach_extra(func: Callable):
        func.__extra_args = config_spec
        func.__auto_cast = auto_cast

        if(hasattr(func, "__doc__")):
            extra_doc = "\n        ".join(
                f" - {name} (Type: {caster}, Default: {default}): {desc}" for name, (default, caster, desc) in config_spec.items()
            )
            func.__doc__ = func.__doc__.format(extra_cli_args=extra_doc)

        return func
    return attach_extra



