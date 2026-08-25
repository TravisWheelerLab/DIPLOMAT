import pandas as pd

import diplomat.processing.type_casters as tc
from diplomat.utils.lazy_import import resolve_lazy_imports

from .sleap_imports import sleap_io


@resolve_lazy_imports
@tc.typecaster_function
def _sleap_nn_tracks_to_diplomat_table(path: tc.PathLike) -> pd.DataFrame:
    labels = sleap_io.load_file(str(path))

    if not isinstance(labels, sleap_io.Labels):
        raise ValueError(
            f"Invalid file passed, only can convert labels, passed {type(labels)}"
        )

    arr = labels.numpy(return_confidence=True)
    track_names = [t.name for t in labels.tracks] or [
        f"track_{i}" for i in range(arr.shape[1])
    ]
    bp_names = [n.name for n in labels.skeletons[0].nodes]
    n_frames = arr.shape[0]
    header = pd.MultiIndex.from_product(
        [track_names, bp_names, ["x", "y", "likelihood"]]
    )
    return pd.DataFrame(arr.reshape(n_frames, -1), columns=header)
