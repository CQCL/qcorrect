import builtins
import inspect
from collections.abc import Callable
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    ParamSpec,
    TypeVar,
    cast,
    dataclass_transform,
)

from guppylang.decorator import custom_guppy_decorator, get_calling_frame
from guppylang.defs import GuppyDefinition, GuppyFunctionDefinition
from guppylang_internals.compiler.core import CompilerContext
from guppylang_internals.definition.common import DefId
from guppylang_internals.definition.custom import OpCompiler
from guppylang_internals.engine import DEF_STORE, ENGINE
from guppylang_internals.tys.subst import Inst
from guppylang_internals.tys.ty import FuncInput, FunctionType, Type
from hugr import ext as he
from hugr import ops
from hugr import tys as ht
from hugr.ext import Extension, OpDef, OpDefSig
from hugr.package import Package
from pydantic_extra_types.semantic_version import SemanticVersion

from qcorrect.lowerable import lowerable_function
from qcorrect.tys import InnerStructType, RawInnerStructDef

if TYPE_CHECKING:
    from guppylang_internals.definition.custom import RawCustomFunctionDef
    from guppylang_internals.definition.function import ParsedFunctionDef

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")


@custom_guppy_decorator
def block(cls: builtins.type[T]) -> builtins.type[T]:
    """Decorator to annotate a class as a code block."""
    defn = RawInnerStructDef(DefId.fresh(), cls.__name__, None, cls)
    DEF_STORE.register_def(defn, get_calling_frame())
    for val in cls.__dict__.values():
        if isinstance(val, GuppyDefinition):
            DEF_STORE.register_impl(defn.id, val.wrapped.name, val.id)
    # Prior to Python 3.13, the `__firstlineno__` attribute on classes is not set.
    # However, we need this information to precisely look up the source for the
    # class later. If it's not there, we can set it from the calling frame:
    if not hasattr(cls, "__firstlineno__"):
        cls.__firstlineno__ = get_calling_frame().f_lineno  # type: ignore[attr-defined]
    # We're pretending to return the class unchanged, but in fact we return
    # a `GuppyDefinition` that handles the comptime logic
    return GuppyDefinition(defn)  # type: ignore[return-value]


@custom_guppy_decorator
def operation(op: GuppyDefinition) -> Callable[[Callable[..., F]], F]:
    """Decorator to annotate functions as operations of the code"""

    def dec(defn: Callable[..., F]) -> F:
        defn.__setattr__("__qct_op__", True)
        defn.__setattr__("__tket_op__", op)

        return defn  #  type: ignore[return-value]

    return dec


def op_dec(
    op_def: OpDef,
) -> Callable[[ht.FunctionType, Inst, CompilerContext], ops.DataflowOp]:
    """Helper function to generate hugr op definition."""

    def op(ty: ht.FunctionType, op_inst: Inst, ctx: CompilerContext) -> ops.DataflowOp:
        return ops.ExtOp(op_def, ty, [arg.to_hugr(ctx) for arg in op_inst])

    return op


@dataclass_transform(kw_only_default=True)
class CodeDefinition(ModuleType, Generic[P]):
    guppy_module: ModuleType
    hugr_ext: Extension
    compiled_defs: ClassVar[dict[str, tuple[DefId, Package]]] = {}

    def __init__(self, *args: P.args, **kwargs: P.kwargs):
        # Set name and docs for Module
        self.__name__ = self.__class__.__name__
        self.__doc__ = self.__class__.__doc__

        # Get class annotations
        cls_annotations = inspect.get_annotations(self.__class__)

        # Set class attributes to replicate `dataclass` behaviour
        for name, type in cls_annotations.items():
            # Check if name in kwargs otherwise use default
            arg_value = kwargs[name] if name in kwargs else self.__getattribute__(name)

            if isinstance(arg_value, type):
                setattr(self, name, arg_value)
            else:
                raise TypeError(f"Expected type {type}")

        # Create new Hugr extension
        self.hugr_ext = Extension(self.__name__, SemanticVersion(0, 1, 0))

        # Add all definitions to module
        for name, defn in inspect.getmembers(self.__class__):
            if hasattr(defn, "__qct_op__"):
                self.__setattr__(name, defn(self))

        # Compile all Guppy definitions
        for name, defn in inspect.getmembers(self):
            if isinstance(defn, GuppyFunctionDefinition):
                compiled_hugr = defn.compile_function()
                # Update FuncDefn name
                for module in compiled_hugr.modules:
                    for _, data in module.nodes():
                        if (
                            isinstance(data.op, ops.FuncDefn)
                            and data.op.f_name == defn.wrapped.name
                        ):
                            data.op.f_name = name
                self.compiled_defs[name] = (defn.id, compiled_hugr)

        # Replace `inner` operations with `outer`
        for name, (
            defn_id,
            compiled_hugr,
        ) in self.compiled_defs.items():
            if isinstance(defn, GuppyDefinition):
                parsed_defn = cast("ParsedFunctionDef", ENGINE.get_parsed(defn_id))

                hugr_func_defn = next(
                    data.op
                    for node, data in compiled_hugr.modules[0].nodes()
                    if isinstance(data.op, ops.FuncDefn) and data.op.f_name == name
                )

                # Define hugr OpDef for outer definition
                outer_op_def = OpDef(
                    name=name,
                    description="",
                    signature=OpDefSig(poly_func=hugr_func_defn.signature),
                    lower_funcs=[
                        he.FixedHugr(
                            ht.ExtensionSet(
                                [ext.name for ext in compiled_hugr.extensions]
                            ),
                            hugr_module,
                        )
                        for hugr_module in compiled_hugr.modules
                    ],
                )

                # Add op_def to hugr extension
                self.hugr_ext.add_op_def(outer_op_def)

                raw_defn = cast("RawCustomFunctionDef", DEF_STORE.raw_defs[defn_id])

                py_func = raw_defn.python_func

                guppy_op = lowerable_function(
                    OpCompiler(op_dec(outer_op_def)),
                    name=name,
                    signature=self.define_outer_type_sig(parsed_defn.ty),
                )(py_func)

                self.__setattr__(name, guppy_op)

    def define_outer_type_sig(self, ty: FunctionType) -> FunctionType:
        """Define a new FunctionType that will be the signature of the `outer`
        operations.

        The function loops through all inputs/output types and replaces any
        InnerStructTypes with outer type definitions. All other types are unchanged."""

        outer_inputs = []

        for f_input in ty.inputs:
            if isinstance(f_input.ty, InnerStructType):
                outer_type = f_input.ty.outer_type
                self.hugr_ext.add_type_def(f_input.ty.hugr_type_def)

                outer_inputs.append(FuncInput(ty=outer_type, flags=f_input.flags))
            else:
                outer_inputs.append(f_input)

        if isinstance(ty.output, InnerStructType):
            outer_type = ty.output.outer_type
            if ty.output.hugr_type_def.name not in self.hugr_ext.types:
                self.hugr_ext.add_type_def(ty.output.hugr_type_def)
            outer_output = cast("Type", outer_type)
        else:
            outer_output = ty.output
        return FunctionType(
            inputs=outer_inputs,
            output=outer_output,
            input_names=ty.input_names,
            params=ty.params,
            comptime_args=ty.comptime_args,
        )

    def lower(self, package: Package) -> None:
        """Function to lower from `outer` operations to `inner`.

        Any `outer` operations are replaced with calls to function definitions defined
        in the hugr extension from the code.
        """

        # Find all nodes to replace
        nodes_to_replace = [
            (node, data, data.op.op_def().name)
            for node, data in package.modules[0].nodes()
            if isinstance(data.op, ops.ExtOp)
            and data.op.op_def().name in self.hugr_ext.operations
        ]

        # Add all lowering functions
        func_defn_node = {}

        for f_name, op in self.hugr_ext.operations.items():
            for lower_funcs in op.lower_funcs:
                lower_hugr = lower_funcs.hugr

                # Delete the module root and change to function definition
                assert isinstance(lower_hugr[lower_hugr.module_root].op, ops.Module)
                assert lower_hugr[lower_hugr.module_root].metadata["name"] == "__main__"

                lower_hugr.entrypoint = lower_hugr.module_root

                # Find node for the function definition
                try:
                    func_node = next(
                        node
                        for node, data in lower_hugr.nodes()
                        if isinstance(data.op, ops.FuncDefn)
                        and data.op.f_name == f_name
                    )
                except StopIteration as e:
                    raise NameError(
                        f"Function Definition ({f_name}) not found in hugr."
                    ) from e

                defn_nodes = package.modules[0].insert_hugr(
                    lower_hugr, package.modules[0].module_root
                )

                # Get new node locations
                func_defn_node[f_name] = defn_nodes[func_node]
                new_module_entry = defn_nodes[lower_hugr.module_root]

                # Update parent/children
                for node in list(package.modules[0][new_module_entry].children):
                    package.modules[0][node].parent = package.modules[0].module_root
                    package.modules[0][package.modules[0].module_root].children.append(
                        node
                    )
                    package.modules[0][new_module_entry].children.remove(node)

                # Delete original entrypoint
                package.modules[0].delete_node(new_module_entry)

        # Replace all outer ops with calls to lowering functions
        for node, data, op_name in nodes_to_replace:
            op_sig = self.hugr_ext.get_op(op_name).signature.poly_func
            op_ports_in = [port for _, port in package.modules[0].incoming_links(node)]
            op_ports_out = [port for _, port in package.modules[0].outgoing_links(node)]

            assert isinstance(op_sig, ht.PolyFuncType)

            # Define call operation
            func_call = ops.Call(signature=op_sig, instantiation=op_sig.body)

            # Remove outer node
            parent = package.modules[0][node].parent
            if parent:
                package.modules[0][parent].children.remove(node)

            _, package.modules[0]._nodes[node.idx] = (
                package.modules[0]._nodes[node.idx],
                None,
            )

            # Free up the metadata dictionary
            node._metadata = {}

            package.modules[0]._free_nodes.append(node)

            call_node = package.modules[0].add_node(
                func_call, data.parent, len(op_ports_out)
            )

            package.modules[0].add_link(
                func_defn_node[op_name].out(0),
                call_node.inp(func_call._function_port_offset()),
            )

            # Link nodes
            for i, out_ports in enumerate(op_ports_in):
                for p_out in out_ports:
                    package.modules[0].delete_link(p_out, node.inp(i))
                    package.modules[0].add_link(p_out, call_node.inp(i))
            for i, in_ports in enumerate(op_ports_out):
                for p_in in in_ports:
                    package.modules[0].delete_link(node.out(i), p_in)
                    package.modules[0].add_link(call_node.out(i), p_in)

    def encode(self, hugr: Package) -> Package:
        """Method to encode a hugr using code operations

        Args:
            hugr: program to be encoded
        """

        # Loop through hugr operations
        # Replace operations in program with code `outer` operations

        return hugr
