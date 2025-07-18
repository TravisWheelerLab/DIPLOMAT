import builtins
import functools
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import Callable, Optional
import warnings
import os


ImportFunction = Callable[[str, Optional[str]], ModuleType]
""" Represents a import function. Must be able to import an object from it's dot notated access path. """


def _simple_import(name: str, pkg: Optional[str] = None) -> ModuleType:
    path = name.split(".")

    if len(path) == 1:
        return import_module(name, pkg)
    else:
        try:
            mod = import_module(".".join(path[:-1]), pkg)
        except ImportError:
            mod = _simple_import(".".join(path[:-1]), pkg)
        return getattr(mod, path[-1])


def _dummy_print(*args, **kwargs):
    pass


class SilentImports:
    def __init__(self):
        self._true_print = None
        self._debug_mode = os.environ.get("DIPLOMAT_DEBUG", False)

    def __enter__(self):
        if not self._debug_mode:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow as tf

            tf.get_logger().setLevel("ERROR")

            warnings.filterwarnings("ignore")
            self._true_print = builtins.print
            builtins.print = _dummy_print

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._debug_mode:
            builtins.print = self._true_print
            del os.environ["TF_CPP_MIN_LOG_LEVEL"]


def _silent_import(name: str, pkg: Optional[str] = None) -> ModuleType:
    with warnings.catch_warnings():
        with SilentImports():
            return _simple_import(name, pkg)


class _OnnxPreloadImport:
    def __init__(self):
        self._onnx_preload = True

    def __call__(self, name: str, pkg: Optional[str] = None) -> ModuleType:
        mod = _simple_import(name, pkg)
        # If first import of onnx and it's the most recent gpu versions, preload needed nvidia dlls...
        if self._onnx_preload and mod.__name__.startswith("onnxruntime"):
            self._onnx_preload = False
            import onnxruntime as ort

            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        return mod


onnx_preload_import = _OnnxPreloadImport()


class ImportFunctions:
    """
    A set of import functions that can be used by the lazy importer object. Includes the following import functions:
     - SIMPLE: A basic implementation of an import function.
     - SILENT: Makes imported package silent by disabling printing and log outputs before importing modules.
    """

    SIMPLE = _simple_import
    SILENT = _silent_import


def verify_existence_of(name: str):
    """
    Verifies a package exists without importing it, otherwise throws an ImportError.

    :param name: The name of the package to test for existence.

    :raises: ImportError if the provided module can't be found.
    """
    if len(name.split(".")) > 1:
        raise ValueError(
            "Can only check top-level modules without attempting to import them."
        )

    try:
        spec = find_spec(name)
        if spec is None:
            raise ImportError(f"Unable to find package '{name}'.")
    except Exception as e:
        raise ImportError(str(e))


def resolve_lazy_imports(func: Callable) -> Callable:
    # Optimization: Lookup lazy imports for this module ahead of time...
    func_module = func.__globals__
    if not "__lazy_imports" in func_module:
        func_module["__lazy_imports"] = [
            k for k, v in func_module.items() if (isinstance(v, LazyImporter))
        ]

    @functools.wraps(func)
    def do_resolution(*args, **kwargs):
        func_globals = func.__globals__
        lazy_imports = func_globals.get("__lazy_imports", None)
        lazy_imports = [] if lazy_imports is None else lazy_imports
        while len(lazy_imports) > 0:
            imp_key = lazy_imports.pop()
            v = func_globals.get(imp_key, None)
            if isinstance(v, LazyImporter):
                func_globals[imp_key] = v.force_import()

        return func(*args, **kwargs)

    return do_resolution


class LazyImporter:
    """
    Represents a Lazily Imported Object. It waits until the user requests functionality before actually importing
    a module.
    """

    NOTHING = object()

    def __init__(
        self,
        name: str,
        pkg: Optional[str] = None,
        import_function: ImportFunction = ImportFunctions.SIMPLE,
    ):
        """
        Create a new lazily import module, object, or function.

        :param name: The name or path of the package or module to import.
        :param pkg: The relative path information needed for relative import. Defaults to None.
        :param import_function: The function to use for importing modules. See ImportFunctions for a list of
                                pre-provided importers.
        """
        self._name = name
        self._pkg = pkg
        self._mod = self.NOTHING
        self._imp = import_function

    def __getattr__(self, item: str) -> "LazyImporter":
        """
        For lazy importers, getting an attribute simply returns another lazy importer.
        """
        return type(self)(f"{self._name}.{item}")

    def __call__(self, *args, **kwargs):
        """
        Calling an attribute of a LazyImporter triggers import of the module.
        """
        if self._mod is self.NOTHING:
            self._mod = self._imp(self._name, self._pkg)

        return self._mod(*args, **kwargs)

    def force_import(self):
        if self._mod is self.NOTHING:
            self._mod = self._imp(self._name, self._pkg)
        return self._mod

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return self._name
