from .frame_pass import FramePass, ConfigError
from typing import List, Any, Dict, Mapping, Iterable


class FramePassBuilder:
    def __init__(self, name: str, config: Dict[str, Any], debug: bool = False):
        supported_passes = {c.get_name(): c for c in FramePass.get_subclasses()}

        self._clazz = supported_passes.get(name, None)
        if self._clazz is None:
            raise ConfigError(f"{name} is not a supported or known pass type!")

        conf_ops = self._clazz.get_config_options()
        for val in ["DEBUG", "debug"]:
            if val in conf_ops and val not in config:
                config[val] = debug

        self._config = config
        self._debug = False

    @property
    def clazz(self) -> FramePass:
        return self._clazz

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def __call__(self, width: int, height: int, thread_count: int = 1) -> FramePass:
        return self._clazz(width, height, thread_count, self._config)

    @classmethod
    def sanitize_pass_config_list(cls, passes: List[Any], debug: bool = False) -> List["FramePassBuilder"]:
        new_passes = []

        for f_pass in passes:
            if isinstance(f_pass, str):
                new_passes.append(cls(f_pass, {}, debug))
            elif isinstance(f_pass, Iterable):
                name, *extra = list(f_pass)
                if not isinstance(name, str):
                    raise ConfigError(
                        f"First argument: '{repr(name)}' is not a string!"
                    )

                config = (
                    dict(extra[0])
                    if ((len(extra) > 0) and isinstance(extra[0], Mapping))
                    else {}
                )
                new_passes.append(cls(name, config, debug))

        return new_passes
