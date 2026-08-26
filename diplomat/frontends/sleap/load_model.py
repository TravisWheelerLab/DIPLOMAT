import numpy as np

import diplomat.processing.type_casters as tc
from diplomat.frontends import ModelInfo, ModelLike
from diplomat.utils.cli_tools import Flag
from diplomat.utils.lazy_import import resolve_lazy_imports

from .sleap_imports import torch


def _resolve_device(gpu_index, use_cpu) -> str:
    if use_cpu:
        return "cpu"
    if gpu_index is not None:
        return f"cuda:{gpu_index}"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@resolve_lazy_imports
@tc.typecaster_function
def load_models(
    config: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    batch_size: tc.Optional[int] = None,
    num_outputs: tc.Optional[int] = None,
    gpu_index: tc.Optional[int] = None,
    refinement_kernel_size: int = 5,
    use_cpu: Flag = False,
) -> tc.Tuple[ModelInfo, ModelLike]:
    device = _resolve_device(gpu_index, bool(use_cpu))
    provider = PredictorExtractor(configs, device, refinement_kernel_size)
    meta = provider.get_metadata()
    if batch_size is None:
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
            frontend="sleap",
        ),
        provider.extract,
    )
