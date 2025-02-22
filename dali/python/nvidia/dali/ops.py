# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#pylint: disable=no-member
import sys
import copy
from itertools import count
from nvidia.dali import backend as b
from nvidia.dali.edge import EdgeReference
from nvidia.dali.types import _type_name_convert_to_string, _type_convert_value, DALIDataType
from nvidia.dali.pipeline import Pipeline
from future.utils import with_metaclass

_cpu_ops = set({})
_gpu_ops = set({})
_mixed_ops = set({})
_support_ops = set({})

def _docstring_generator(cls):
    op_name = cls.__name__
    op_dev = []
    if op_name in _cpu_ops:
        op_dev.append("'CPU'")
    if op_name in _gpu_ops:
        op_dev.append("'GPU'")
    if op_name in _mixed_ops:
        op_dev.append("'mixed'")
    if op_name in _support_ops:
        op_dev.append("'support'")
    pre_doc = "This is a " + ", ".join(op_dev) + " operator\n\n"

    schema = b.GetSchema(op_name)
    # insert tag to easily link to the operator
    ret = '.. _' + op_name + ':\n\n'
    ret += pre_doc
    ret += schema.Dox()
    ret += '\n'
    if schema.IsSequenceOperator():
        ret += "\nThis operator expects sequence inputs\n"
    elif schema.AllowsSequences():
        ret += "\nThis operator allows sequence inputs\n"

    if schema.IsDeprecated():
        use_instead = schema.DeprecatedInFavorOf()
        ret += "\n.. warning::\n\n   This operator is now deprecated"
        if use_instead:
            ret +=". Use `" + use_instead + "` instead"
        ret += "\n"

    if schema.IsNoPrune():
        ret += "\nThis operator will **not** be optimized out of the graph.\n"

    ret += """
Parameters
----------
"""
    for arg in schema.GetArgumentNames():
        dtype = schema.GetArgumentType(arg)
        arg_name_doc = "`" + arg + "` : "
        ret += (arg_name_doc +
                _type_name_convert_to_string(dtype, schema.IsTensorArgument(arg)))
        if schema.IsArgumentOptional(arg):
            default_value_string = schema.GetArgumentDefaultValueString(arg)
            # Evaluating empty string results in an error
            # so we need to prevent that
            if default_value_string:
                default_value = eval(default_value_string)
            else:
                default_value = default_value_string
            if dtype == DALIDataType.STRING:
                default_value = "\'" + str(default_value) + "\'"
            ret += (", optional, default = " +
                    str(_type_convert_value(dtype, default_value)))
        indent = '\n' + " " * len(arg_name_doc)
        ret += indent
        ret += schema.GetArgumentDox(arg).replace("\n", indent)
        ret += '\n'
    return ret

class _OpCounter(object):
    #pylint: disable=too-few-public-methods
    _op_count = count(0)
    def __init__(self):
        self._id = next(self._op_count)

    @property
    def id(self):
        return self._id

class _OperatorInstance(object):
    def __init__(self, inputs, op, **kwargs):
        self._counter = _OpCounter()
        self._inputs = inputs
        self._outputs = []
        self._op = op
        self._spec = op.spec.copy()
        if "name" in kwargs.keys():
            self._name = kwargs["name"]
        else:
            self._name = '__' + type(op).__name__ + "_" + str(self._counter.id)
        # Add inputs
        if inputs:
            if isinstance(inputs[0], EdgeReference):
                for inp in inputs:
                    if not isinstance(inp, EdgeReference):
                        raise TypeError(
                            ("Expected inputs of type " +
                            "EdgeReference. Received " +
                            "input type {}.")
                            .format(type(inp).__name__))
                    self._spec.AddInput(inp.name, inp.device)
            elif isinstance(inputs[0], list):
                length = len(inputs[0])
                for i in range(length):
                    for inp in inputs:
                        if not isinstance(inp, list):
                            raise TypeError(
                                ("Expected inputs of type list of " +
                                "EdgeReference. Received " +
                                "input type {}.")
                                .format(type(inp).__name__))
                        if len(inp) != length:
                            raise RuntimeError(
                                    ("Expected input lists " +
                                    "to have the same length " +
                                    "({}). Received list of " +
                                    "length {}.")
                                    .format(length, len(inp)))
                        if not isinstance(inp[i], EdgeReference):
                            raise TypeError(
                                ("Expected inputs of type " +
                                "EdgeReference. Received " +
                                "input type {}.")
                                .format(type(inp[i]).__name__))
                        self._spec.AddInput(inp[i].name, inp[i].device)
                self._spec.AddArg("num_input_sets", length)
            else:
                raise TypeError(
                    ("Expected inputs of type EdgeReference or list of " +
                    "EdgeReference. Received input type {}")
                    .format(type(inputs[0]).__name__))
        # Argument inputs
        for k in sorted(kwargs.keys()):
            if k not in ["name"]:
                if not isinstance(kwargs[k], EdgeReference):
                    raise TypeError(
                            ("Expected inputs of type " +
                            "EdgeReference. Received " +
                            "input type {}")
                            .format(type(kwargs[k]).__name__))
                self._spec.AddArgumentInput(k, kwargs[k].name)
                self._inputs = list(self._inputs) + [kwargs[k]]

        if self._op.schema.IsDeprecated():
            use_instead = self._op.schema.DeprecatedInFavorOf()
            msg = "WARNING: `{}` is now deprecated".format(type(self._op).__name__)
            if use_instead:
                msg +=". Use `" + use_instead + "` instead"
            print(msg)

    def check_args(self):
        self._op.schema.CheckArgs(self._spec)

    def generate_outputs(self):
        pipeline = Pipeline.current()
        # Add outputs
        if self._op.device == "gpu" or self._op.device == "mixed":
            output_device = "gpu"
        else:
            output_device = "cpu"

        num_output = self._op.schema.CalculateOutputs(self._spec) + self._op.schema.CalculateAdditionalOutputs(self._spec)

        if num_output == 0 and self._op.preserve:
            t_name = type(self._op).__name__ + "_id_" + str(self.id) + "_sink"
            pipeline.add_sink(EdgeReference(t_name, output_device, self))
            return

        for i in range(num_output):
            t_name = type(self._op).__name__ + "_id_" + str(self.id) + "_output_" + str(i)
            t = EdgeReference(t_name, output_device, self)
            self._spec.AddOutput(t.name, t.device)
            if self._op.preserve:
                pipeline.add_sink(t)
            self.append_output(t)

    @property
    def id(self):
        return self._counter.id

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def spec(self):
        return self._spec

    @property
    def name(self):
        return self._name

    def append_output(self, output):
        self._outputs.append(output)

class _DaliOperatorMeta(type):
    @property
    def __doc__(self):
        return _docstring_generator(self)

def python_op_factory(name, op_device = "cpu"):
    class Operator(with_metaclass(_DaliOperatorMeta, object)):
        def __init__(self, **kwargs):
            self._spec = b.OpSpec(type(self).__name__)
            self._schema = b.GetSchema(type(self).__name__)

            # Get the device argument. We will need this to determine
            # the device that our outputs will be stored on
            if "device" in kwargs.keys():
                self._device = kwargs["device"]
                del kwargs["device"]
            else:
                self._device = op_device
            self._spec.AddArg("device", self._device)

            if "preserve" in kwargs.keys():
                self._preserve = kwargs["preserve"]
            else:
                self._preserve = False
            self._spec.AddArg("preserve", self._preserve)
            self._preserve = self._preserve or self._schema.IsNoPrune()

            # Store the specified arguments
            for key, value in kwargs.items():
                if isinstance(value, list):
                    if not value:
                        raise RuntimeError("List arguments need to have at least 1 element.")
                dtype = self._schema.GetArgumentType(key)
                converted_value = _type_convert_value(dtype, value)
                self._spec.AddArg(key, converted_value)

        @property
        def spec(self):
            return self._spec

        @property
        def schema(self):
            return self._schema

        @property
        def device(self):
            return self._device

        @property
        def preserve(self):
            return self._preserve

        def __call__(self, *inputs, **kwargs):
            if (len(inputs) > self._schema.MaxNumInput() or
                    len(inputs) < self._schema.MinNumInput()):
                raise ValueError(
                    ("Operator {} expects [{}, " +
                    "{}] inputs, but received {}")
                    .format(type(self).__name__,
                            self._schema.MinNumInput(),
                            self._schema.MaxNumInput(),
                            len(inputs)))

            op_instance = _OperatorInstance(inputs, self, **kwargs)
            op_instance.generate_outputs()

            if len(op_instance.outputs) == 1:
                return op_instance.outputs[0]
            return op_instance.outputs

    Operator.__name__ = str(name)
    return Operator

def _load_ops():
    global _cpu_ops
    global _gpu_ops
    global _mixed_ops
    global _support_ops
    _cpu_ops = _cpu_ops.union(set(b.RegisteredCPUOps()))
    _gpu_ops = _gpu_ops.union(set(b.RegisteredGPUOps()))
    _mixed_ops = _mixed_ops.union(set(b.RegisteredMixedOps()))
    _cpu_gpu_ops = _cpu_ops.union(_gpu_ops).union(_mixed_ops)
    _support_ops = _support_ops.union(set(b.RegisteredSupportOps()))
    for op_name in _cpu_gpu_ops:
        if not hasattr(sys.modules[__name__], op_name):
            setattr(sys.modules[__name__], op_name,
                    python_op_factory(op_name, op_device = "cpu"))
    # add support ops
    for op_name in _support_ops:
        if not hasattr(sys.modules[__name__], op_name):
            setattr(sys.modules[__name__], op_name,
                    python_op_factory(op_name, op_device = "support"))
_load_ops()

def Reload():
    _load_ops()

# custom wrappers around ops
class TFRecordReader(with_metaclass(_DaliOperatorMeta, object)):
    global _cpu_ops
    _cpu_ops = _cpu_ops.union({'TFRecordReader'})

    def __init__(self, path, index_path, features, **kwargs):
        if isinstance(path, list):
            self._path = path
        else:
            self._path = [path]
        if isinstance(index_path, list):
            self._index_path = index_path
        else:
            self._index_path = [index_path]
        self._schema = b.GetSchema("_TFRecordReader")
        self._spec = b.OpSpec("_TFRecordReader")
        self._device = "cpu"

        self._spec.AddArg("path", self._path)
        self._spec.AddArg("index_path", self._index_path)

        for key, value in kwargs.items():
            self._spec.AddArg(key, value)

        self._features = features

    @property
    def spec(self):
        return self._spec

    @property
    def schema(self):
        return self._schema

    @property
    def device(self):
        return self._device

    def __call__(self, *inputs, **kwargs):
        if (len(inputs) > self._schema.MaxNumInput() or
                len(inputs) < self._schema.MinNumInput()):
            raise ValueError(
                ("Operator {} expects [{}, " +
                "{}] inputs, but received {}")
                .format(type(self).__name__,
                        self._schema.MinNumInput(),
                        self._schema.MaxNumInput(),
                        len(inputs)))

        op_instance = _OperatorInstance(inputs, self, **kwargs)
        outputs = {}
        feature_names = []
        features = []
        for i, (feature_name, feature) in enumerate(self._features.items()):
            t_name = "_TFRecordReader" + "_id_" + str(op_instance.id) + "_output_" + str(i)
            t = EdgeReference(t_name, self._device, op_instance)
            op_instance.spec.AddOutput(t.name, t.device)
            op_instance.append_output(t)
            outputs[feature_name] = t
            feature_names.append(feature_name)
            features.append(feature)

        op_instance.spec.AddArg("feature_names", feature_names)
        op_instance.spec.AddArg("features", features)
        return outputs

class PythonFunction(with_metaclass(_DaliOperatorMeta, object)):
    global _cpu_ops
    _cpu_ops = _cpu_ops.union({'PythonFunction'})

    def __init__(self, function, num_outputs=1, **kwargs):
        self._schema = b.GetSchema("PythonFunctionImpl")
        self._spec = b.OpSpec("PythonFunctionImpl")
        self._device = "cpu"

        for key, value in kwargs.items():
            self._spec.AddArg(key, value)

        self.function = function
        self.num_outputs = num_outputs
        self._preserve = True

    @property
    def spec(self):
        return self._spec

    @property
    def schema(self):
        return self._schema

    @property
    def device(self):
        return self._device

    @property
    def preserve(self):
        return self._preserve

    def __call__(self, *inputs, **kwargs):
        pipeline = Pipeline.current()
        if (len(inputs) > self._schema.MaxNumInput() or
                len(inputs) < self._schema.MinNumInput()):
            raise ValueError(
                ("Operator {} expects [{}, " +
                 "{}] inputs, but received {}")
                .format(type(self).__name__,
                        self._schema.MinNumInput(),
                        self._schema.MaxNumInput(),
                        len(inputs)))

        op_instance = _OperatorInstance(inputs, self, **kwargs)
        op_instance.spec.AddArg("function_id", id(self.function))
        op_instance.spec.AddArg("num_outputs", self.num_outputs)
        if self.num_outputs == 0:
            t_name = "PythonFunctionImpl" + "_id_" + str(op_instance.id) + "_sink"
            t = EdgeReference(t_name, self._device, op_instance)
            pipeline.add_sink(t)
            return
        outputs = []
        for i in range(self.num_outputs):
            t_name = "PythonFunctionImpl" + "_id_" + str(op_instance.id) + "_output_" + str(i)
            t = EdgeReference(t_name, self._device, op_instance)
            op_instance.spec.AddOutput(t.name, t.device)
            op_instance.append_output(t)
            pipeline.add_sink(t)
            outputs.append(t)
        return outputs[0] if len(outputs) == 1 else outputs

def cpu_ops():
    return _cpu_ops

def gpu_ops():
    return _gpu_ops

def support_ops():
    return _support_ops

def mixed_ops():
    return _mixed_ops

def register_cpu_op(name):
    global _cpu_ops
    _cpu_ops = _cpu_ops.union({name})
