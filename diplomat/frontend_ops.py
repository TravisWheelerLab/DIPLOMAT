import inspect
from diplomat.processing.type_casters import typecaster_function
from diplomat.utils.pretty_printer import printer as print
from diplomat.utils.cli_tools import get_summary_from_doc_str
from dataclasses import asdict


@typecaster_function
def list_all_frontends():
    """
    List all frontends currently included with this version of DIPLOMAT.
    """
    from diplomat import _FRONTENDS

    print(f"Number of frontends: {len(_FRONTENDS)}\n\n")

    for frontend in _FRONTENDS:
        print(f"Frontend '{frontend.get_package_name()}':")
        print("Description:")
        print(f"\t{inspect.getdoc(frontend)}\n")


@typecaster_function
def list_loaded_frontends():
    """
    List frontends that have successfully loaded, and list their supported functions.
    """
    from diplomat import _LOADED_FRONTENDS, _FRONTENDS

    frontend_docs = {f.get_package_name(): inspect.getdoc(f) for f in _FRONTENDS}

    print(f"Number of loaded frontends: {len(_LOADED_FRONTENDS)}\n")

    for name, funcs in _LOADED_FRONTENDS.items():
        print(f"Loaded Frontend '{name}':")
        print("Description:")
        print(f"\t{frontend_docs[name]}")
        print("Supported Functions:")
        for k, v in funcs:
            if(k.startswith("_")):
                continue
            print(f"\t{k}")
            print("\t\t" + " ".join(get_summary_from_doc_str(str(inspect.getdoc(v))).split()))
        print("\n")