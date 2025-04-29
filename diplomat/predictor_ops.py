from diplomat import processing
from diplomat.processing.type_casters import typecaster_function, Union, List, get_type_name, Optional
from diplomat.utils.pretty_printer import printer as print
from diplomat.utils.cli_tools import Flag, positional_argument_count


@typecaster_function
def list_predictor_plugins():
    """
    Retrieve and print all currently installed and available predictor plugins that can be used with deeplabcut to
    the console...

    :return: Nothing, if one wants to get the plugins for other code look at processing.get_predictor_plugins().
    """
    # Load the plugins...
    predictors = processing.get_predictor_plugins()

    for predictor in predictors:
        print(f"Plugin Name: '{predictor.get_name()}'")
        print("Description: ")
        print("\t", predictor.get_description())
        print()

def _get_predictor_settings(predictor_name: Union[List[str], str, None] = None):
    """
    Returns the available/modifiable settings for a specified predictor plugin.

    :param predictor_name: The string or list of strings being the names of the predictor plugins to view customizable
                      settings for. If None, will print settings for all currently available predictors.
                      Defaults to None.

    :return: ConfigSpec, a dictionary relating each predictor setting to a 3-tuple of its default value, type, and description strnig.
    """
    from typing import Iterable

    # Convert whatever the predictor_name argument is to a list of predictor plugins
    if predictor_name is None:
        predictors = processing.get_predictor_plugins()
    elif isinstance(predictor_name, str):
        predictors = [processing.get_predictor(predictor_name)]
    elif isinstance(predictor_name, Iterable):
        predictors = [processing.get_predictor(name) for name in predictor_name]
    else:
        raise ValueError(
            "Argument 'predictor_name' not of type Iterable[str], string, or None!!!"
        )

    # yield name and settings for each plugin.
    for predictor in predictors:
        plugin_name = predictor.get_name()
        config_spec = predictor.get_settings()
        if config_spec is None:
            yield (plugin_name, {})
        else:
            yield (plugin_name, config_spec)

@typecaster_function
@positional_argument_count(1)
def get_predictor_settings(predictor: Optional[Union[List[str], str]] = None):
    """
    Prints the available/modifiable settings for a specified predictor plugin.

    :param predictor: The string or list of strings being the names of the predictor plugins to view customizable
                      settings for. If None, will print settings for all currently available predictors.
                      Defaults to None.

    :return: Nothing, prints to console....
    """

    # Print name, and settings for each plugin.
    for (plugin_name, config_spec) in _get_predictor_settings(predictor):
        print(f"Plugin Name: {plugin_name}")
        print("Arguments: ")
        if config_spec is {}:
            print("None")
        else:
            for name, (def_val, val_type, desc) in config_spec.items():
                print(f"Name: '{name}'")
                print(f"Type: {get_type_name(val_type)}")
                print(f"Default Value: {def_val}")
                print(f"Description: \n\t{desc}\n")

        print()


@typecaster_function
@positional_argument_count(1)
def test_predictor_plugin(predictor: Optional[Union[List[str], str]] = None, interactive: Flag = False):
    """
    Run the tests for a predictor plugin.

    :param predictor: The name of the predictor or to run tests for, or a list of names of the predictors to run.
                      If the predictor_name is not specified or set to None, then run tests for all the
                      predictor plugins...
    :param interactive: A boolean. If True, the program will wait for user input after every test, to allow the user
                        to easily read tests one by one... If false, all tests will be run at once with no user
                        interaction. Defaults to false.
    :return: Nothing, prints test info to console...
    """
    from typing import Iterable
    import traceback

    # Convert whatever the predictor_name argument is to a list of predictor plugins
    if predictor is None:
        predictors = processing.get_predictor_plugins()
    elif isinstance(predictor, str):
        predictors = [processing.get_predictor(predictor)]
    elif isinstance(predictor, Iterable):
        predictors = [processing.get_predictor(name) for name in predictor]
    else:
        raise ValueError(
            "Argument 'predictor_name' not of type Iterable[str], string, or None!!!"
        )

    # Test plugins by calling there tests...
    for predictor in predictors:
        print(f"Testing Plugin: '{predictor.get_name()}'")
        # Get the tests...
        tests = predictor.get_tests()

        # If this test contains no test, let the user know and move to the next plugin.
        if tests is None:
            print(f"Plugin {predictor.get_name()} has no tests...\n")
            print()
            continue

        passed_tests = 0
        total_tests = 0

        # Iterate tests printing there results...
        for test_meth in tests:
            print(f"Running Test '{test_meth.__name__}':")
            try:
                passed, expected, actual = test_meth()

                print(f"Results: {'Passed' if passed else 'Failed'}")
                if passed:
                    passed_tests += 1
                if not passed:
                    print(f"Expected Results: {expected}")
                    print(f"Actual Results: {actual}")

            except Exception as excep:
                print("Results: Failed With Exception")
                traceback.print_exception(type(excep), excep, excep.__traceback__)
            finally:
                total_tests += 1
                print()
                if interactive:
                    input("Press Enter To Continue: ")
                    print()

        passing_percent = 100 * (passed_tests / total_tests)
        print(
            f"RESULTS: {passed_tests} out of {total_tests} passed, passing rate of {passing_percent:2.2f}%\n"
        )
        print()
