import os
import sys
from diplomat.processing.type_casters import typecaster_function, PathLike, Union, Optional, List, Dict, Any, get_typecaster_annotations
from diplomat.utils.pretty_printer import printer as print
from diplomat.utils.cli_tools import func_to_command, allow_arbitrary_flags, Flag
from argparse import ArgumentParser
import typing
from types import ModuleType
from diplomat.utils.tweak_ui import UIImportError


def _get_casted_args(tc_func, extra_args):
    """
    PRIVATE: Get correctly casted extra arguments for the provided typecasting function. Any arguments that don't match those in the function
             get thrown out.
    """
    def_tcs = get_typecaster_annotations(tc_func)
    extra = getattr(tc_func, "__extra_args", {})
    autocast = getattr(tc_func, "__auto_cast", True)

    new_args = {}

    for k, v in extra_args.items():
        if(k in def_tcs):
            new_args[k] = def_tcs[k](v)
        elif(k in extra):
            new_args[k] = extra[k][1](v) if(autocast) else v
        else:
            print(f"Warning: command '{tc_func.__name__}' does not have an argument called '{k}', ignoring the argument...")

    return new_args

def _find_frontend(config: os.PathLike, **kwargs: typing.Any) -> typing.Tuple[str, ModuleType]:
    from diplomat import _LOADED_FRONTENDS

    for name, funcs in _LOADED_FRONTENDS.items():
        if(funcs._verifier(
            config=config,
            **kwargs
        )):
            print(f"Frontend '{name}' selected.")
            return (name, funcs)

    print("Could not find a frontend that correctly handles the passed config and other arguments. Make sure the config passed is valid.")
    sys.exit(1)

def _display_help(
    frontend_name: str,
    method_type: str,
    calling_command_name: str,
    command_func: typing.Callable,
    is_cli: bool
):
    if(is_cli):
        print(f"\n\nHelp for {frontend_name}'s {method_type} command:\n")
        func_to_command(command_func, ArgumentParser(prog=calling_command_name)).print_help()
    else:
        import pydoc
        help_dumper = pydoc.Helper(output=sys.stdout, input=sys.stdin)

        print(f"\n\nDocstring for {frontend_name}'s {method_type} method:\n")
        help_dumper.help(command_func)


@allow_arbitrary_flags
@typecaster_function
def track(
    config: Union[List[PathLike], PathLike],
    videos: Optional[Union[List[PathLike], PathLike]] = None,
    frame_stores: Optional[Union[List[PathLike], PathLike]] = None,
    num_outputs: Optional[int] = None,
    predictor: Optional[str] = None,
    predictor_settings: Optional[Dict[str, Any]] = None,
    help_extra: Flag = False,
    **extra_args
):
    """
    Run DIPLOMAT tracking on videos and/or frame stores. Automatically select a frontend based on the passed arguments.

    :param config: The path to the configuration file for the project. The format of this argument will depend on the frontend.
    :param videos: A single path or list of paths to video files to run analysis on.
    :param frame_stores: A single path or list of paths to frame store files to run analysis on.
    :param num_outputs: An integer, the number of bodies to track in the video. Defaults to 1.
    :param predictor: An optional string, specifying the predictor plugin to make predictions with. You can get a list of all available predictors
                      and descriptions using the "diplomat predictors list" command or "diplomat.list_predictor_plugins" function.
    :param predictor_settings: An optional dictionary, listing the settings to use for the specified predictor plugin instead of the defaults.
                               If not specified, the frontend will determine the settings in a frontend specific manner. To see the settings a
                               predictor plugin supports, use the "diplomat predictors list_settings" command or "diplomat.get_predictor_settings"
                               function. To get more information about how a frontend gets settings if not passed, set the help_extra parameter
                               to True to print additional settings for the selected frontend instead of running tracking.
    :param help_extra: Boolean, if set to true print extra settings for the automatically selected frontend instead of running tracking.
    :param extra_args: Any additional arguments (if the CLI, flags starting with '--') are passed to the automatically selected frontend.
                       To see valid values, run track with extra_help flag set to true. Extra arguments that are not found in the frontend
                       analysis function are thrown out.
    """
    from diplomat import CLI_RUN

    selected_frontend_name, selected_frontend = _find_frontend(
        config=config,
        videos=videos,
        frame_stores=frame_stores,
        num_outputs=num_outputs,
        predictor=predictor,
        predictor_settings=predictor_settings,
        **extra_args
    )

    if(help_extra):
        _display_help(selected_frontend_name, "video analysis", "diplomat track", selected_frontend.analyze_videos, CLI_RUN)
        _display_help(selected_frontend_name, "frame analysis", "diplomat track", selected_frontend.analyze_frames, CLI_RUN)
        return

    if(videos is None and frame_stores is None):
        print("No frame stores or videos passed, terminating.")
        return

    # If some videos are supplied, run the frontends video analysis function.
    if(videos is not None):
        print("Running on videos...")
        selected_frontend.analyze_videos(
            config=config,
            videos=videos,
            num_outputs=num_outputs,
            predictor=predictor,
            predictor_settings=predictor_settings,
            **_get_casted_args(selected_frontend.analyze_videos, extra_args)
        )

    # If some frame stores are supplied, run the frontends frame analysis function.
    if(frame_stores is not None):
        print("Running on frame stores...")
        selected_frontend.analyze_frames(
            config=config,
            frame_stores=frame_stores,
            num_outputs=num_outputs,
            predictor=predictor,
            predictor_settings=predictor_settings,
            **_get_casted_args(selected_frontend.analyze_frames, extra_args)
        )

@allow_arbitrary_flags
@typecaster_function
def unsupervised(
    config: Union[List[PathLike], PathLike],
    videos: Optional[Union[List[PathLike], PathLike]] = None,
    frame_stores: Optional[Union[List[PathLike], PathLike]] = None,
    num_outputs: Optional[int] = None,
    settings: Optional[Dict[str, Any]] = None,
    help_extra: Flag = False,
    **extra_args
):
    """
    Run diplomat in unsupervised mode on the specified config and videos or frame stores. An alias for
    DIPLOMAT track with the SegmentedFramePassEngine predictor.

    :param config: The path to the configuration file for the project. The format of this argument will depend on the frontend.
    :param videos: A single path or list of paths to video files to run analysis on.
    :param frame_stores: A single path or list of paths to frame store files to run analysis on.
    :param num_outputs: An integer, the number of bodies to track in the video. Defaults to 1.
    :param settings: An optional dictionary, listing the settings to use for the SegmentedFramePassEngine predictor plugin.
                     If not specified, the frontend will determine the settings in a frontend specific manner. To see the settings the
                     SegmentedFramePassEngine supports, use the "diplomat predictors list_settings -p SegmentedFramePassEngine" command
                     or "diplomat.get_predictor_settings('SegmentedFramePassEngine')". To get more information about how a frontend gets
                     settings if not passed, set the help_extra parameter to True to print additional settings for the selected
                     frontend instead of running tracking.
    :param help_extra: Boolean, if set to true print extra settings for the automatically selected frontend instead of running tracking.
    :param extra_args: Any additional arguments (if the CLI, flags starting with '--') are passed to the automatically selected frontend.
                       To see valid values, run track with extra_help flag set to true. Extra arguments that are not found in the frontend
                       analysis function are thrown out.
    """
    track(
        config=config,
        videos=videos,
        frame_stores=frame_stores,
        num_outputs=num_outputs,
        predictor="SegmentedFramePassEngine",
        predictor_settings=settings,
        help_extra=help_extra,
        **extra_args
    )


@allow_arbitrary_flags
@typecaster_function
def supervised(
    config: Union[List[PathLike], PathLike],
    videos: Optional[Union[List[PathLike], PathLike]] = None,
    frame_stores: Optional[Union[List[PathLike], PathLike]] = None,
    num_outputs: Optional[int] = None,
    settings: Optional[Dict[str, Any]] = None,
    help_extra: Flag = False,
    **extra_args
):
    """
    Run diplomat in supervised mode on the specified config and videos or frame stores. An alias for
    DIPLOMAT track with the SupervisedSegmentedFramePassEngine predictor.

    :param config: The path to the configuration file for the project. The format of this argument will depend on the frontend.
    :param videos: A single path or list of paths to video files to run analysis on.
    :param frame_stores: A single path or list of paths to frame store files to run analysis on.
    :param num_outputs: An integer, the number of bodies to track in the video. Defaults to 1.
    :param settings: An optional dictionary, listing the settings to use for the SupervisedSegmentedFramePassEngine predictor plugin.
                     If not specified, the frontend will determine the settings in a frontend specific manner. To see the settings the
                     SupervisedSegmentedFramePassEngine supports, use the "diplomat predictors list_settings -p SupervisedSegmentedFramePassEngine"
                     command or "diplomat.get_predictor_settings('SupervisedSegmentedFramePassEngine')". To get more information about how a
                     frontend gets settings if not passed, set the help_extra parameter to True to print additional settings for the selected
                     frontend instead of running tracking.
    :param help_extra: Boolean, if set to true print extra settings for the automatically selected frontend instead of running tracking.
    :param extra_args: Any additional arguments (if the CLI, flags starting with '--') are passed to the automatically selected frontend.
                       To see valid values, run track with extra_help flag set to true. Extra arguments that are not found in the frontend
                       analysis function are thrown out.
    """
    track(
        config=config,
        videos=videos,
        frame_stores=frame_stores,
        num_outputs=num_outputs,
        predictor="SupervisedSegmentedFramePassEngine",
        predictor_settings=settings,
        help_extra=help_extra,
        **extra_args
    )


@allow_arbitrary_flags
@typecaster_function
def annotate(
    config: Union[List[PathLike], PathLike],
    videos: Optional[Union[List[PathLike], PathLike]] = None,
    help_extra: Flag = False,
    **extra_args
):
    """
    Have diplomat annotate, or label a video given it has already been tracked. Automatically searches for the
    correct frontend to do labeling based on the passed config argument.

    :param config: The path to the configuration file for the project. The format of this argument will depend on the frontend.
    :param videos: A single path or list of paths to video files run annotation on.
    :param help_extra: Boolean, if set to true print extra settings for the automatically selected frontend instead of running video annotation.
    :param extra_args: Any additional arguments (if the CLI, flags starting with '--') are passed to the automatically selected frontend.
                       To see valid values, run annotate with extra_help flag set to true. Extra arguments that are not found in the frontend
                       analysis function are thrown out.
    """
    from diplomat import CLI_RUN

    # Iterate the frontends, looking for one that actually matches our request...
    selected_frontend_name, selected_frontend = _find_frontend(config=config, videos=videos, **extra_args)

    if(help_extra):
        _display_help(selected_frontend_name, "video labeling", "diplomat annotate", selected_frontend.label_videos, CLI_RUN)
        return

    if(videos is None):
        print("No videos passed, terminating.")
        return

    selected_frontend.label_videos(
        config=config,
        videos=videos,
        **_get_casted_args(selected_frontend.label_videos, extra_args)
    )


@allow_arbitrary_flags
@typecaster_function
def tweak(
    config: Union[List[PathLike], PathLike],
    videos: Optional[Union[List[PathLike], PathLike]] = None,
    help_extra: Flag = False,
    **extra_args
):
    """
    Make modifications to DIPLOMAT produced tracking results created for a video using a limited version supervised labeling UI. Allows for touching
    up and fixing any minor issues that may arise after tracking and saving results.

    :param config: The path to the configuration file for the project. The format of this argument will depend on the frontend.
    :param videos: A single path or list of paths to video files to tweak the tracks of.
    :param help_extra: Boolean, if set to true print extra settings for the automatically selected frontend instead of showing the UI.
    :param extra_args: Any additional arguments (if the CLI, flags starting with '--') are passed to the automatically selected frontend.
                       To see valid values, run tweak with extra_help flag set to true. Extra arguments that are not found in the frontend
                       tweak function are thrown out.
    """
    from diplomat import CLI_RUN

    selected_frontend_name, selected_frontend = _find_frontend(config=config, videos=videos, **extra_args)

    if(help_extra):
        _display_help(selected_frontend_name, "label tweaking", "diplomat tweak", selected_frontend.tweak_videos, CLI_RUN)
        return

    if(videos is None):
        print("No videos passed, terminating.")
        return

    try:
        selected_frontend.tweak_videos(
            config=config,
            videos=videos,
            **_get_casted_args(selected_frontend.tweak_videos, extra_args)
        )
    except UIImportError as e:
        print(e)