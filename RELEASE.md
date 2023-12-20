# Release Instructions

DIPLOMAT is setup to be easy to release, using Github-Actions. To release a new DIPLOMAT version, the following
steps can be followed.
1. Update `RELEASE_NOTES.md` to list the latest features in DIPLOMAT.
2. Modify DIPLOMAT's `__init__.py` file (located at `diplomat/__init__.py`) to increase the version number.
   This is done by modifying the `__version__` variable in the file to a new semver, such as `"1.0.1"`.
3. Commit your changes, and push them to the `main` branch. This can be done by:
    - Manually pushing to the main branch.
    - Creating, and merging a GitHub Pull-Request.

Once done, the GitHub action hooks will automatically detect the new version change, and create a new pypi version
and release on GitHub (including a tag for the version).
