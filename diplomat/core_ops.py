import sys
from diplomat.processing.type_casters import typecaster_function, PathLike, Union, Optional, List, Dict, Any, get_typecaster_annotations
from diplomat.utils.pretty_printer import printer as print
from diplomat.utils.cli_tools import func_to_command, allow_arbitrary_flags, Flag
from argparse import ArgumentParser

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

    return new_args


@allow_arbitrary_flags
@typecaster_function
def track(
    config: PathLike,
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
                       To see valid values, run track with --extra_help=True. Extra arguments that are not found in the frontend
                       analysis function are thrown out.
    """
    from diplomat import _LOADED_FRONTENDS, CLI_RUN

    selected_frontend_name = None
    selected_frontend = None

    # Iterate the frontends, looking for one that actually matches our request...
    for name, funcs in _LOADED_FRONTENDS.items():
        if(funcs._verifier(
            config=config,
            videos=videos,
            frame_stores=frame_stores,
            num_outputs=num_outputs,
            predictor=predictor,
            predictor_settings=predictor_settings,
            **extra_args
        )):
            selected_frontend_name = name
            selected_frontend = funcs
            break

    if(selected_frontend_name is None):
        print("Could not find a frontend that correctly handles the passed config and other arguments. Make sure the config passed is valid.")
        return

    print(f"Frontend '{selected_frontend_name}' selected.")

    if(help_extra):
        if(CLI_RUN):
            print(f"Help for 'DIPLOMAT {selected_frontend_name}' frontend:")
            print(f"\n\nHelp for video analysis command:\n\n")
            func_to_command(selected_frontend.analyze_videos, ArgumentParser(prog="diplomat track")).print_help()
            print(f"\n\nHelp for frame analysis command:\n\n")
            func_to_command(selected_frontend.analyze_frames, ArgumentParser(prog="diplomat track")).print_help()
        else:
            import pydoc
            help_dumper = pydoc.Helper(output=sys.stdout, input=sys.stdin)

            print(f"Help for 'DIPLOMAT {selected_frontend_name}' frontend:")
            print(f"\n\nDocstring for video analysis command:\n\n")
            help_dumper.help(selected_frontend.analyze_videos)
            print(f"\n\nDocstring for frame analysis command:\n\n")
            help_dumper.help(selected_frontend.analyze_frames)
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
    config: PathLike,
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
                       To see valid values, run track with --extra_help=True. Extra arguments that are not found in the frontend
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
    config: PathLike,
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
                       To see valid values, run track with --extra_help=True. Extra arguments that are not found in the frontend
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
        extra_args=extra_args
    )


@allow_arbitrary_flags
@typecaster_function
def annotate(
    config: PathLike,
    videos: Optional[Union[List[PathLike], PathLike]] = None,
    help_extra: Flag = False,
    **extra_args
):
    """
    Have diplomat annotate, or label a video given it has already been tracked. Automatically searches for the
    correct frontend to do labeling based on the passed config argument.

    :param config: The path to the configuration file for the project. The format of this argument will depend on the frontend.
    :param videos: A single path or list of paths to video files run annotation on.
    :param help_extra: Boolean, if set to true print extra settings for the automatically selected frontend instead of running tracking.
    :param extra_args: Any additional arguments (if the CLI, flags starting with '--') are passed to the automatically selected frontend.
                       To see valid values, run track with --extra_help=True. Extra arguments that are not found in the frontend
                       analysis function are thrown out.
    """
    from diplomat import _LOADED_FRONTENDS, CLI_RUN

    selected_frontend_name = None
    selected_frontend = None

    # Iterate the frontends, looking for one that actually matches our request...
    for name, funcs in _LOADED_FRONTENDS.items():
        if(funcs._verifier(config=config, videos=videos, **extra_args)):
            selected_frontend_name = name
            selected_frontend = funcs
            break

    if(selected_frontend_name is None):
        print("Could not find a frontend that correctly handles the passed config and other arguments. Make sure the config passed is valid.")
        return

    print(f"Frontend '{selected_frontend_name}' selected.")

    if(help_extra):
        if(CLI_RUN):
            print(f"Help for 'DIPLOMAT {selected_frontend_name}' frontend:")
            print(f"Help for video labeling command:\n\n")
            func_to_command(selected_frontend.label_videos, ArgumentParser(prog="diplomat annotate")).print_help()
        else:
            import pydoc
            help_dumper = pydoc.Helper(output=sys.stdout, input=sys.stdin)

            print(f"Help for 'DIPLOMAT {selected_frontend_name}' frontend:")
            print(f"Docstring for video analysis command:\n\n")
            help_dumper.help(selected_frontend.label_videos)
        return

    if(videos is None):
        print("No videos passed, terminating.")
        return

    selected_frontend.label_videos(
        config=config,
        videos=videos,
        **_get_casted_args(selected_frontend.label_videos, extra_args)
    )