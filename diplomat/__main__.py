import sys

import diplomat
from diplomat.utils.cli_tools import build_full_parser
from argparse import ArgumentParser
from dataclasses import asdict

function_tree = {
    "predictors": {
        "__description": "Contains subcommands for listing, testing, and printing information "
                         "for the currently installed predictor plugins in this version of DIPLOMAT.",
        "list": diplomat.list_predictor_plugins,
        "test": diplomat.test_predictor_plugin,
        "list_settings": diplomat.get_predictor_settings
    },
    "track": diplomat.track,
    "supervised": diplomat.supervised,
    "unsupervised": diplomat.unsupervised,
    "annotate": diplomat.annotate,
    "split_videos": diplomat.split_videos,
    "frontends": {
        "__description": "Contains subcommands for listing available frontends and inspecting the functions each frontend supports.",
        "list": {
            "__description": "List DIPLOMAT frontends and their descriptions.",
            "all": diplomat.list_all_frontends,
            "loaded": diplomat.list_loaded_frontends
        }
    }
}

for frontend_name, funcs in diplomat._LOADED_FRONTENDS.items():
    frontend_commands = {
        name: func for name, func in asdict(funcs).items() if(not name.startswith("_"))
    }

    doc_str = getattr(getattr(diplomat, frontend_name), "__doc__", None)
    if(doc_str is not None):
        frontend_commands["__description"] = doc_str

    function_tree[frontend_name] = frontend_commands


parser = build_full_parser(
    function_tree,
    ArgumentParser(prog="DIPLOMAT", description="A tool for multi-animal tracking.")
)
diplomat.CLI_RUN = True
parser(sys.argv[1:])
