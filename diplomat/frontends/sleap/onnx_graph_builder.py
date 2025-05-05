from typing import Iterable, Optional, Union, List
from .sleap_imports import onnx


class OnnxVar:
    def __init__(self, name: str, elem_type: onnx.TensorProto.DataType, shape: Iterable[Optional[int]], producing_op: Optional["_OnnxOpOutput"] = None):
        self.name = name
        self.elem_type = elem_type
        self.shape = tuple(shape)
        self.producing_op = producing_op


class _OnnxOpOutput:
    def __init__(self, op: "OnnxOp", index: int):
        self.op = op
        self.index = index
        self._name_override = None

    def to_var(self, name: str, elem_type: onnx.TensorProto.DataType, shape: Iterable[Optional[int]]) -> OnnxVar:
        self._name_override = name
        return OnnxVar(name, elem_type, shape, self)

    @property
    def name(self) -> str:
        if(self._name_override is not None):
            return self._name_override
        return f"{self.op.name}:{self.index}"


class OnnxOp:

    def __init__(
        self,
        op_type: str,
        *inputs: Union[OnnxVar, _OnnxOpOutput, "OnnxOp", None],
        doc_string: str = None,
        domain: str = None,
        overload: str = None,
        **attributes
    ):
        self.op_type = op_type
        self.doc_string = doc_string
        self.domain = domain
        self.overload = overload
        self.name = None  # Set later...

        self.inputs = []
        for inp in inputs:
            if isinstance(inp, OnnxOp):
                inp = inp[0]
            if isinstance(inp, (OnnxVar, _OnnxOpOutput)):
                self.inputs.append(inp)
            elif inp is None:
                self.inputs.append(inp)
            else:
                raise ValueError(f"Received invalid input: {inp} of type: {type(inp)}.")
        self.outputs = []
        self.attributes = attributes

    def __getitem__(self, idx: int) -> _OnnxOpOutput:
        if not isinstance(idx, int):
            raise ValueError("Only can access outputs by index!")

        for next_idx in range(len(self.outputs), idx + 1):
            self.outputs.append(_OnnxOpOutput(self, next_idx))

        return self.outputs[idx]

    def to_var(self, name: str, elem_type: onnx.TensorProto.DataType, shape: Iterable[Optional[int]]) -> OnnxVar:
        return self[0].to_var(name, elem_type, shape)


def _topo_sort(node: Union[_OnnxOpOutput, OnnxVar, None], visit_list: list, visited: set):
    if node is None:
        return

    if isinstance(node, OnnxVar):
        if node not in visited:
            visited.add(node)
            visit_list.append(node)
        return

    op = node.op

    if op in visited:
        return

    visited.add(op)

    for inps in op.inputs:
        _topo_sort(inps, visit_list, visited)

    visit_list.append(op)


def to_onnx_graph_def(
    name: str,
    outputs: List[OnnxVar],
):
    visit_list = []
    visited = set()

    for out in outputs:
        if(out.producing_op is None):
            raise ValueError("Model has output variable with no connections to the graph!")
        _topo_sort(out.producing_op, visit_list, visited)

    op_counts = {}
    onnx_nodes = []
    implicit_inputs = []

    # Traverse the nodes in order now...
    for node in visit_list:
        if isinstance(node, OnnxVar) and node.producing_op is None:
            implicit_inputs.append(onnx.helper.make_tensor_value_info(
                node.name,
                node.elem_type,
                node.shape
            ))
        elif isinstance(node, OnnxOp):
            # Set the name...
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
            count = op_counts[node.op_type]
            node.name = f"{node.op_type}_{count}"

            op_inputs = [n.name if n is not None else "" for n in node.inputs]
            op_outputs = [n.name for n in node.outputs]

            onnx_nodes.append(onnx.helper.make_node(
                node.op_type,
                op_inputs,
                op_outputs,
                node.name,
                node.doc_string,
                node.domain,
                node.overload,
                **node.attributes
            ))
        else:
            raise ValueError(f"Unrecognized node {node} of type {type(node)}.")

    onnx_outputs = [
        onnx.helper.make_tensor_value_info(n.name, n.elem_type, n.shape) for n in outputs
    ]

    return onnx.helper.make_graph(
        onnx_nodes,
        name,
        implicit_inputs,
        onnx_outputs
    )
