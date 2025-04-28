from typing import Optional

import numpy as np

import diplomat.processing.type_casters as tc
from diplomat.frontends import ModelInfo, ModelLike
from diplomat.frontends.sleap.run_utils import _load_configs, _dict_get_path
from diplomat.frontends.sleap.sleap_providers import PredictorExtractor
from diplomat.utils.cli_tools import Flag
import onnxruntime as ort


def _build_provider_ordering(device_index: Optional[int], use_cpu: bool):
    supported_devices = ort.get_available_providers()
    device_config = []

    if(not use_cpu):
        if("CUDAExecutionProvider" in supported_devices):
            device_config.append(("CUDAExecutionProvider", {"device_id": device_index}))
        if("ROCMExecutionProvider" in supported_devices):
            device_config.append(("ROCMExecutionProvider", {"device_id": device_index}))
        if("CoreMLExecutionProvider" in supported_devices):
            device_config.append("CoreMLExecutionProvider")

    # Fallback...
    device_config.append("CPUExecutionProvider")
    return device_config


@tc.typecaster_function
def load_models(
    config: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    batch_size: tc.Optional[int] = None,
    num_outputs: tc.Optional[int] = None,
    gpu_index: tc.Optional[int] = None,
    output_suffix: str = "",
    refinement_kernel_size: int = 5,
    use_cpu: Flag = False,
) -> tc.Tuple[ModelInfo, ModelLike]:
    configs = _load_configs(config)

    provider = PredictorExtractor(
        configs,
        refinement_kernel_size,
        providers=_build_provider_ordering(gpu_index, bool(use_cpu))
    )

    meta = provider.get_metadata()

    if(batch_size is None):
        batch_size = meta["batch_size"]

    return (
        ModelInfo(
            num_outputs=num_outputs,
            batch_size=batch_size,
            dotsize=int(np.ceil(meta["sigma"] / meta["input_scaling"])),
            colormap=None,
            shape_list=None,
            alphavalue=0.7,
            pcutoff=0.1,
            line_thickness=1,
            bp_names=meta["bp_names"],
            skeleton=meta["skeleton"],
            frontend="sleap"
        ),
        provider.extract
    )

