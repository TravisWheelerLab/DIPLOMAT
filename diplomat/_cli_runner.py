import sys
import diplomat
from diplomat.utils.cli_tools import build_full_parser
from argparse import ArgumentParser
from dataclasses import asdict


def get_static_cli_tree() -> dict:
    return {
        "_category": "track",
        "predictors": {
            "__description": "Contains subcommands for listing, testing, and printing information "
                             "for the currently installed predictor plugins in this version of DIPLOMAT.",
            "_category": "dev",
            "list": diplomat.list_predictor_plugins,
            "test": diplomat.test_predictor_plugin,
            "list_settings": diplomat.get_predictor_settings
        },
        "track_with": diplomat.track_with,
        "track_and_interact": diplomat.track_and_interact,
        "track": diplomat.track,
        "annotate": diplomat.annotate,
        "split_videos": diplomat.split_videos,
        "tweak": diplomat.tweak,
        "yaml": diplomat.yaml,
        "convert": diplomat.convert,
        "interact": diplomat.interact,
        "frontends": {
            "__description": "Contains subcommands for listing available frontends and inspecting the functions "
                             "each frontend supports.",
            "_category": "dev",
            "list": {
                "__description": "List DIPLOMAT frontends and their descriptions.",
                "all": diplomat.list_all_frontends,
                "loaded": diplomat.list_loaded_frontends
            }
        }
    }


def get_dynamic_cli_tree() -> dict:
    function_tree = get_static_cli_tree()

    for frontend_name, funcs in diplomat._LOADED_FRONTENDS.items():
        frontend_commands = {
            name: func for name, func in funcs if(not name.startswith("_"))
        }

        doc_str = getattr(getattr(diplomat, frontend_name), "__doc__", None)
        if(doc_str is not None):
            frontend_commands["__description"] = doc_str

        frontend_commands["_category"] = "frontend"
        function_tree[frontend_name] = frontend_commands

    return function_tree


def main():
    function_tree = get_dynamic_cli_tree()

    parser = ArgumentParser(prog="DIPLOMAT", description="A tool for multi-animal tracking.")
    parser.add_argument("--version", "-v", action="version", version=f"%(prog)s {diplomat.__version__}")
    parser = build_full_parser(
        function_tree,
        parser
    )

    diplomat.CLI_RUN = True
    parser(sys.argv[1:])
