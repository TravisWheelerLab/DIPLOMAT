from typing import Type, Optional, Tuple, List
from sphinx.application import Sphinx, Config
from docutils import nodes
from sphinx import addnodes
from pathlib import Path

from sphinx.domains.python import PyClasslike, PythonDomain, ObjType, PyXRefRole, PyAttribute

from diplomat.predictors.fpe.sparse_storage import AttributeDict
from diplomat.processing.type_casters import get_type_name
from diplomat.utils.pluginloader import load_plugin_classes

import diplomat.predictors as predictors
from diplomat.processing import Predictor, ConfigSpec
import diplomat

import diplomat.predictors.fpe.frame_passes as frame_passes
from diplomat.predictors.fpe.frame_pass import FramePass


class PyPlugin(PyClasslike):
    def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
        res = super().get_index_text(modname, name_cls)
        return res.replace("class", "plugin")

class PyOption(PyAttribute):
    def get_signature_prefix(self, sig: str) -> List[nodes.Node]:
        return [nodes.Text("option"), addnodes.desc_sig_space()]

    def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
        name, cls = name_cls

        clsname, attrname = name.rsplit('.', 1)
        if modname and self.env.config.add_module_names:
            clsname = '.'.join([modname, clsname])

        return 'Option %s (in plugin %s)' % (attrname, clsname)


def register_custom_py_types():
    # Add support for plugin type...
    PythonDomain.object_types["plugin"] = ObjType("plugin", 'plugin', 'class', 'exc', 'obj')
    PythonDomain.directives["plugin"] = PyPlugin
    PythonDomain.roles["plugin"] = PyXRefRole()

    # Add support for option type (based on data or attribute type)...
    PythonDomain.object_types["option"] = ObjType("option", 'option', 'attr', 'obj')
    PythonDomain.directives["option"] = PyOption
    PythonDomain.roles["option"] = PyXRefRole()


_BUILD_LOC = "api/_autosummary"

templates = {
    "option": "option-template.rst",
    "plugin": "plugin-template.rst",
    "api": "api-template.rst"
}

def load_templates(src: Path):
    for k in templates:
        with ((src / "_templates") / templates[k]).open("r") as f:
            templates[k] = f.read()

def format_settings(settings: Optional[ConfigSpec]) -> str:
    if(settings is None):
        return "    This plugin can't be passed any options."

    string_list = []

    for name, (default, caster, desc) in settings.items():
        string_list.append(templates["option"].format(
            name=name,
            type=get_type_name(caster),
            default=repr(default),
            desc=desc
        ))

    return "\n".join(string_list)


def get_predictor_rst(plugin: Type[Predictor]) -> str:
    return templates["plugin"].format(
        name = plugin.get_name(),
        title_eqs="=" * len(plugin.get_name()),
        plugin_type=Predictor.__module__ + "." + Predictor.__name__,
        plugin_type_name = Predictor.__name__,
        desc = plugin.get_description(),
        settings = format_settings(plugin.get_settings())
    )


def get_frame_pass_rst(plugin: Type[FramePass]) -> str:
    desc = getattr(plugin, "__doc__", "")
    desc = desc if(desc is not None) else ""

    return templates["plugin"].format(
        name = plugin.get_name(),
        title_eqs="=" * len(plugin.get_name()),
        plugin_type=FramePass.__module__ + "." + FramePass.__name__,
        plugin_type_name = FramePass.__name__,
        desc = desc,
        settings = format_settings(plugin.get_config_options())
    )


def document_predictor_plugins(path: Path) -> list:
    api_list = []

    for plugin in load_plugin_classes(predictors, Predictor):
        dest = path / ("diplomat.predictors." + plugin.get_name() + ".rst")
        api_list.append(("diplomat.predictors." + plugin.get_name(), plugin.get_description()))
        dest.parent.mkdir(exist_ok=True)

        print(f"\tWriting {dest.name}...")

        with dest.open("w") as f:
            f.write(get_predictor_rst(plugin))

    return api_list


def document_frame_pass_plugins(path: Path) -> list:
    api_list = []

    for plugin in load_plugin_classes(frame_passes, FramePass):
        doc_str = plugin.__doc__

        dest = path / ("diplomat.predictors.frame_passes." + plugin.get_name() + ".rst")
        api_list.append(("diplomat.predictors.frame_passes." + plugin.get_name(), doc_str if(doc_str is not None) else ""))
        dest.parent.mkdir(exist_ok=True)

        print(f"\tWriting {dest.name}...")

        with dest.open("w") as f:
            f.write(get_frame_pass_rst(plugin))

    return api_list


PLUGINS = {
    "predictors": document_predictor_plugins,
    "frame_passes": document_frame_pass_plugins
}

EXTRA = {
    "core": diplomat,
}

FIX_ALL = {
    diplomat.utils,
    diplomat.processing
}

def fix_all_on_module(module):
    from types import ModuleType, FunctionType
    import pkgutil

    if(hasattr(module, "__path__")):
        path = list(iter(module.__path__))[0]

        for importer, name, ispkg in pkgutil.iter_modules([path], module.__name__ + "."):
            attr_name = name.split(".")[-1]

            if(attr_name.startswith("_")):
                continue

            try:
                setattr(module, attr_name, importer.find_module(name).load_module(name))
                val = getattr(module, attr_name)
                fix_all_on_module(val)
            except:
                raise

    all_attr = getattr(module, "__all__", None)
    if(not (all_attr is None or len(all_attr) == 0)):
        return

    module.__all__ = []

    for attr_name in dir(module):
        if(attr_name.startswith("_")):
            continue

        try:
            val = getattr(module, attr_name)

            if(isinstance(val, ModuleType) and val.__name__.startswith(module.__name__)):
                module.__all__.append(attr_name)
            elif(isinstance(val, (type, FunctionType)) and val.__module__.startswith(module.__name__)):
                module.__all__.append(attr_name)
            elif(isinstance(val, object) and type(val).__module__.startswith(module.__name__)):
                module.__all__.append(attr_name)
        except:
            pass

    print(f"New all for {module}: {module.__all__}")


def write_api_rst(api_dir: Path, document_lists: AttributeDict) -> None:
    with (api_dir / "api.rst").open("w") as f:
        f.write(templates["api"].format(diplomat=document_lists))

def on_config_init(app: Sphinx, config: Config) -> None:
    load_templates(Path(app.srcdir))
    register_custom_py_types()

    build_dir = Path(app.srcdir) / _BUILD_LOC
    build_dir.mkdir(parents=True, exist_ok=True)

    document_lists = AttributeDict()
    document_lists.files = AttributeDict()

    for name, documenter in PLUGINS.items():
        print(f"Documenting {name}...")
        file_list = documenter(build_dir)
        document_lists[name] = "\n".join(
            f"    * - :doc:`{file.split('.')[-1]} <_autosummary/{file}>`\n"
            f"      - {doc}" for file, doc in file_list
        )

        document_lists.files[name] = "\n".join(f"    _autosummary/{file}" for file, doc in file_list)


    for name, module in EXTRA.items():
        listing = getattr(module, "__all__", dir(module))
        document_lists[name] = "\n".join(
            f"    ~{getattr(getattr(module, sub_item), '__module__', module.__name__)}.{sub_item}"
            for sub_item in listing if(not sub_item.startswith("_"))
        )

    write_api_rst(build_dir.parent, document_lists)


def setup(app: Sphinx) -> dict:
    app.connect("config-inited", on_config_init)
    for module in FIX_ALL:
        fix_all_on_module(module)

    return {
        "version": "0.0.1"
    }