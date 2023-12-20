Changes in this version of diplomat:
 - The `SegmentedSupervisedFramePassEngine` and `SegmentedFramePassEngine` support saving state to disk using
   `--storage_mode=disk`, and is now the default. This does negatively affect performance on some platforms,
   so a full in memory run can be done by setting the `storage_mode` setting to `"memory"`.
 - New `diplomat restore` command for restoring the UI from `.dipui` files. This requires running predictions with the
   storage mode set to `"disk"`, which is the default now.
 - New installation process with mamba that is more consistent and stable across platforms.