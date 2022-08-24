import sys

import diplomat
from diplomat.utils.cli_tools import build_full_parser
from argparse import ArgumentParser
from dataclasses import asdict

function_tree = {
    "predictors": {
        "list": diplomat.list_predictor_plugins,
        "test": diplomat.test_predictor_plugin,
        "list_settings": diplomat.get_predictor_settings
    }
}

for frontend_name, funcs in diplomat._LOADED_FRONTENDS.items():
    function_tree[frontend_name] = {
        name: func for name, func in asdict(funcs).items()
    }

parser = build_full_parser(function_tree, ArgumentParser(prog="DIPLOMAT", description="A tool for multi-animal tracking."))
parser(sys.argv[1:])
