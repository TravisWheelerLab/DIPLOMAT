from typing import Type, Optional, Tuple, List

from sphinx.application import Sphinx, Config
from docutils import nodes
from sphinx import addnodes
from pathlib import Path

from sphinx.domains.python import PyClasslike, PythonDomain, ObjType, PyAttribute, PyXRefRole
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.mock import mock
from sphinx.roles import XRefRole

MOCK_PACKAGES = ["deeplabcut", "tensorflow", "scipy", "pandas"]

with mock(MOCK_PACKAGES):
    from diplomat.predictors.fpe.sparse_storage import AttributeDict
    from diplomat.processing.type_casters import get_type_name

    import diplomat.predictors as predictors
    from diplomat.processing import Predictor, ConfigSpec
    import diplomat

    import diplomat.predictors.fpe.frame_passes as frame_passes
    from diplomat.predictors.fpe.frame_pass import FramePass
    from diplomat.frontends import DIPLOMATBaselineCommands

def load_plugins_with_mocks(module, clazz):
    from diplomat.utils.pluginloader import load_plugin_classes
    with mock(MOCK_PACKAGES):
        return load_plugin_classes(module, clazz)


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

def patch_python_sphinx_domain():
    def _resolve_xref(self, env: BuildEnvironment, fromdocname, builder, typ, target, node, contnode):
        from sphinx import util

        if(typ == "plugin"):
            doc_path = util.docname_join("", node['reftarget'])

            if doc_path not in env.all_docs:
                return None
            else:
                caption = node.astext() if(node['refexplicit']) else util.nodes.clean_astext(env.titles[doc_path])
                innernode = nodes.strong('', '', nodes.literal(caption, caption, classes=['plugin']))
                return util.nodes.make_refnode(builder, fromdocname, doc_path, None, innernode)
        else:
            return type(self)._old_resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode)

    PythonDomain._old_resolve_xref = PythonDomain.resolve_xref
    PythonDomain.resolve_xref = _resolve_xref


class PluginXRefRole(XRefRole):
    def process_link(
        self,
        env: BuildEnvironment,
        refnode: nodes.Element,
        has_explicit_title: bool,
        title: str,
        target: str
    ) -> Tuple[str, str]:
        title, target = super().process_link(env, refnode, has_explicit_title, title, target)

        if(not has_explicit_title):
            if(target.startswith("~")):
                title = target.lstrip(".~").split(".")[-1]

        target = target.lstrip(".~")

        return title, "/" + _BUILD_LOC.strip("/") + "/" + target


def register_custom_py_types(app: Sphinx):
    # Add support for plugin type...
    PythonDomain.object_types["plugin"] = ObjType("plugin", 'plugin')
    PythonDomain.directives["plugin"] = PyPlugin
    PythonDomain.roles["plugin"] = PluginXRefRole(warn_dangling=True, innernodeclass=nodes.inline)

    # Add support for option type (based on data or attribute type)...
    PythonDomain.object_types["option"] = ObjType("option", 'option', 'attr', 'obj')
    PythonDomain.directives["option"] = PyOption
    PythonDomain.roles["option"] = PyXRefRole()

    patch_python_sphinx_domain()


_BUILD_LOC = "api/_autosummary"

templates = {
    "option": "option-template.rst",
    "plugin": "plugin-template.rst",
    "api": "api-template.rst",
    "frontend": "frontend-template.rst",
    "cli_header": "cli-header-template.rst",
    "cli_entry": "cli-entry-template.rst"
}

def load_templates(src: Path):
    for k in templates:
        with ((src / "_templates") / templates[k]).open("r") as f:
            templates[k] = f.read()


def clean_doc_str(doc: str) -> str:
    return " ".join(doc.strip().split())

def format_settings(settings: Optional[ConfigSpec]) -> str:
    if(settings is None):
        return "    This plugin can't be passed any options."

    string_list = []

    for name, (default, caster, desc) in settings.items():
        desc = getattr(desc, "__sphinx_str__", desc.__str__)()

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
        desc = clean_doc_str(plugin.get_description()),
        settings = format_settings(plugin.get_settings())
    )


def get_frame_pass_rst(plugin: Type[FramePass]) -> str:
    desc = getattr(plugin, "__doc__", "")
    desc = clean_doc_str(desc if(desc is not None) else "")

    return templates["plugin"].format(
        name = plugin.get_name(),
        title_eqs="=" * len(plugin.get_name()),
        plugin_type=FramePass.__module__ + "." + FramePass.__name__,
        plugin_type_name = FramePass.__name__,
        desc = desc,
        settings = format_settings(plugin.get_config_options())
    )

def get_frontend_rst(name: str, methods: DIPLOMATBaselineCommands):
    from dataclasses import asdict

    module_name = "diplomat." + name
    doc = getattr(getattr(diplomat, name), "__doc__", "")
    doc = "" if(doc is None) else doc

    return templates["frontend"].format(
        module_name = module_name,
        module_name_eqs = "=" * len(module_name),
        desc = clean_doc_str(doc),
        function_list = "\n".join(
            f"    ~{func.__module__}.{func.__name__}" for name, func in asdict(methods).items() if(not func.__name__.startswith("_"))
        )
    )


def document_predictor_plugins(path: Path) -> list:
    api_list = []

    for plugin in load_plugins_with_mocks(predictors, Predictor):
        dest = path / ("diplomat.predictors." + plugin.get_name() + ".rst")
        api_list.append(("diplomat.predictors." + plugin.get_name(), clean_doc_str(plugin.get_description())))
        dest.parent.mkdir(exist_ok=True)

        print(f"\tWriting {dest.name}...")

        with dest.open("w") as f:
            f.write(get_predictor_rst(plugin))

    return api_list


def document_frame_pass_plugins(path: Path) -> list:
    api_list = []

    for plugin in load_plugins_with_mocks(frame_passes, FramePass):
        doc_str = plugin.__doc__

        dest = path / ("diplomat.predictors.frame_passes." + plugin.get_name() + ".rst")
        api_list.append(("diplomat.predictors.frame_passes." + plugin.get_name(), clean_doc_str(doc_str if(doc_str is not None) else "")))
        dest.parent.mkdir(exist_ok=True)

        print(f"\tWriting {dest.name}...")

        with dest.open("w") as f:
            f.write(get_frame_pass_rst(plugin))

    return api_list


def document_frontend_plugins(path: Path) -> list:
    api_list = []

    for plug in diplomat._FRONTENDS:
        name = plug.get_package_name()

        if(name not in diplomat._LOADED_FRONTENDS):
            raise ValueError(f"Unable to load frontend '{name}' to build docs! Make sure all packages are installed...")

        methods = diplomat._LOADED_FRONTENDS[name]

        dest = path / ("diplomat." + name + ".rst")
        api_list.append(("diplomat." + name, clean_doc_str(plug.__doc__ if (plug.__doc__ is not None) else "")))
        dest.parent.mkdir(exist_ok=True)

        with dest.open("w") as f:
            f.write(get_frontend_rst(name, methods))

    return api_list


PLUGINS = {
    "predictors": document_predictor_plugins,
    "frame_passes": document_frame_pass_plugins,
    "frontends": document_frontend_plugins
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

def write_api_rst(api_dir: Path, document_lists: AttributeDict) -> None:
    with (api_dir / "api.rst").open("w") as f:
        f.write(templates["api"].format(diplomat=document_lists))


# TODO: Add CLI support...
def write_cli_entry(func_name: str, func) -> str:
    pass

def _write_cli_rst_helper(func_tree: dict, entry_list: list):
    return

def write_cli_rst(api_dir: Path) -> None:
    from diplomat._cli_runner import get_dynamic_cli_tree

    title_under = ["-", "^"]




def on_config_init(app: Sphinx, config: Config) -> None:
    load_templates(Path(app.srcdir))
    register_custom_py_types(app)

    build_dir = Path(app.srcdir) / _BUILD_LOC
    build_dir.mkdir(parents=True, exist_ok=True)

    document_lists = AttributeDict()
    document_lists.files = AttributeDict()
    final_folder_name = Path(_BUILD_LOC).name

    for name, documenter in PLUGINS.items():
        print(f"Documenting {name}...")
        file_list = documenter(build_dir)
        document_lists[name] = "\n".join(
            f"    * - :py:plugin:`~{file}`\n"
            f"      - {doc}" for file, doc in file_list
        )

        document_lists.files[name] = "\n".join(f"    {final_folder_name}/{file}" for file, doc in file_list)


    for name, module in EXTRA.items():
        listing = getattr(module, "__all__", dir(module))
        document_lists[name] = "\n".join(
            f"    ~{getattr(getattr(module, sub_item), '__module__', module.__name__)}.{sub_item}"
            for sub_item in listing if(not sub_item.startswith("_"))
        )

    write_api_rst(build_dir.parent, document_lists)
    write_cli_rst(build_dir.parent.parent)

def setup(app: Sphinx) -> dict:
    app.setup_extension("sphinx.ext.autodoc")

    app.connect("config-inited", on_config_init)

    for module in FIX_ALL:
        fix_all_on_module(module)

    return {
        "version": "0.0.1"
    }