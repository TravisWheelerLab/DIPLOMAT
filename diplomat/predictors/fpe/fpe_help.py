from .frame_pass import FramePass


class FPEString:
    def __init__(self, prefix: str):
        self._prefix = prefix

    def __sphinx_str__(self) -> str:
        return (
            f"{self._prefix} See :ref:`Frame Passes` to see the list of currently supported frame passes and options."
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        from diplomat.processing.type_casters import get_type_name
        desc_lst = []

        for fp in FramePass.get_subclasses():
            desc_lst.append(f"\tPass '{fp.get_name()}' Settings: [[[")
            options = fp.get_config_options()
            if(options is None):
                desc_lst.append("\t\tNo settings available...")
            else:
                for name, (def_val, caster, desc) in options.items():
                    desc_lst.append(f"\t\tSetting Name: '{name}':")
                    desc_lst.append(f"\t\tDefault Value: {def_val}")
                    desc_lst.append(f"\t\tValue Type: {get_type_name(caster)}")
                    desc_lst.append(f"\t\tDescription:\n\t\t\t{desc}\n")

            desc_lst.append("\t]]]\n")

        return (
            f"{self._prefix} The following frame passes and options are currently supported:\n\n"
            + "\n".join(desc_lst)
        )

