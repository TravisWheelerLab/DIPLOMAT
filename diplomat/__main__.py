from argparse import ArgumentParser
import sys

parser = ArgumentParser(description="CLI Interface for DIPLOMAT")
sub_parsers = parser.add_subparsers(required=True)

predictors = sub_parsers.add_parser("predictors", help="Perform operations on predictors.")
sub_pred_commands = predictors.add_subparsers(required=True)
list_predictors = sub_pred_commands.add_parser("list", help="List available predictor plugins included with this version of DIPLOMAT.")


results = parser.parse_args(sys.argv[1:])
print(results)