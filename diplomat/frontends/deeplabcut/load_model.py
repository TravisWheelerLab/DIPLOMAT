from pathlib import Path


def get_model_folder(train_fraction, shuffle, cfg, modelprefix=""):
    Task = cfg["Task"]
    date = cfg["date"]
    iterate = "iteration-" + str(cfg["iteration"])
    return Path(
        modelprefix,
        "dlc-models",
        iterate,
        Taskk
        + date
        + "-trainset"
        + str(int(train_fraction * 100))
        + "shuffle"
        + str(shuffle),
    )

# TODO: DLC model loading...
# part_pred no grad or Adam in it.
# locref_pred no grad or Adam in it (pose/pairwise_pred/block4/BiasAdd)...
def load_tf_model():
    pass
