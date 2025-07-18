import json

import numpy as np
import pandas as pd
import diplomat.processing.type_casters as tc


@tc.typecaster_function
def _sleap_analysis_h5_to_diplomat_table(path: tc.PathLike) -> pd.DataFrame:
    import h5py

    if not h5py.is_hdf5(path):
        raise ValueError("Passed file is not an hdf5 file!")

    with h5py.File(path, "r") as f:
        info = json.loads(f["metadata"].attrs["json"])

        for k in ("videos", "tracks"):
            outside_key = f"{k}_json"
            if outside_key in f:
                info[k] = [json.loads(item) for item in f[outside_key]]

        if len(info["videos"]) < 1:
            raise ValueError("Sleap analysis file must have at least 1 video file...")

        part_names = [n["name"] for n in info["nodes"]]

        frames = f["frames"][:]
        instances = f["instances"][:]
        points = f["points"][:]
        pred_points = f["pred_points"][:]

        # Allocate an array to store all tracks...
        track_names = [name for t_id, name in info["tracks"]]
        track_inst_counts = np.zeros(len(track_names), dtype=np.int64)
        first_video_frames = frames[frames["video"] == 0]
        frame_count = int(np.max(first_video_frames["frame_idx"])) + 1

        tracks = np.zeros(
            (len(track_inst_counts), len(part_names), 3, frame_count), dtype=np.float32
        )

        for frame in first_video_frames:
            frame_idx = frame["frame_idx"]
            sub_instances = instances[
                frame["instance_id_start"] : frame["instance_id_end"]
            ]
            sub_instances = sub_instances[np.argsort(sub_instances["instance_type"])]

            for instance in sub_instances:
                track_idx = instance["track"]
                # Not assigned to first skeleton or no assigned track, skip...
                if track_idx >= len(track_names):
                    continue

                track_inst_counts[track_idx] += 1
                p_ref = points if (instance["instance_type"] == 0) else pred_points
                sub_points = p_ref[
                    instance["point_id_start"] : instance["point_id_end"]
                ]
                scores = sub_points["score"] if "score" in sub_points.dtype.names else 1
                tracks[track_idx, :, 0, frame_idx] = sub_points["x"]
                tracks[track_idx, :, 1, frame_idx] = sub_points["y"]
                tracks[track_idx, :, 2, frame_idx] = scores * sub_points["visible"]

        # Throw away tracks that have no instances...
        track_names = [
            n for i, n in enumerate(track_names) if (track_inst_counts[i] > 0)
        ]
        tracks = tracks[track_inst_counts > 0]

        # Make header...
        header = pd.MultiIndex.from_product(
            [track_names, part_names, ["x", "y", "likelihood"]]
        )
        table = pd.DataFrame(tracks.reshape((-1, frame_count)).T, columns=header)

    return table
