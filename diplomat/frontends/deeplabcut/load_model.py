from pathlib import Path


def get_model_folder(train_fraction, shuffle, cfg, modelprefix=""):
    Task = cfg["Task"]
    date = cfg["date"]
    iterate = "iteration-" + str(cfg["iteration"])
    return Path(
        modelprefix,
        "dlc-models",
        iterate,
        Task
        + date
        + "-trainset"
        + str(int(train_fraction * 100))
        + "shuffle"
        + str(shuffle),
    )