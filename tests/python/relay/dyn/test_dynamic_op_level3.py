# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
""" Support level3 operator test cases.
"""
import numpy as np
import pytest
import tvm
from tvm import te
from tvm import relay
from tvm.relay import create_executor, transform
from tvm.relay.testing import check_grad, run_infer_type
import tvm.testing


def verify_func(func, data, ref_res):
    assert isinstance(data, list)
    for target, ctx in tvm.testing.enabled_targets():
        for kind in ["vm", "debug"]:
            mod = tvm.ir.IRModule.from_expr(func)
            intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
            op_res = intrp.evaluate()(*data)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)
            relay.backend.compile_engine.get().clear()


@tvm.testing.uses_gpu
def test_dyn_reshape():
    def verify_reshape(shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType((len(newshape),), "int64"))
        z = relay.reshape(x, y)

        func = relay.Function([x, y], z)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        x_data = np.ones(shape).astype("float32")
        ref_res = np.reshape(x_data, oshape)
        check_grad(
            run_infer_type(func),
            inputs=[x_data, np.array(newshape).astype("int64")],
            test_inputs=[x_data],
            eps=1e-3,
        )
        verify_func(func, [x_data, np.array(newshape).astype("int64")], ref_res)

    verify_reshape((2, 3, 4), (8, 3), (8, 3))
    verify_reshape((4, 7), (2, 7, 2), (2, 7, 2))
    verify_reshape((2, 3, 4), (4, 0, 2), (4, 3, 2))
    verify_reshape((2, 3, 4), (2, 0, 0), (2, 3, 4))
    verify_reshape((2, 3, 4), (0, -1), (2, 12))
    verify_reshape((2, 3, 4), (-1, 0), (8, 3))
    verify_reshape((2, 3, 4), (-3, 4), (6, 4))
    verify_reshape((2, 3, 4, 5), (-3, -3), (6, 20))
    verify_reshape((2, 3, 4), (0, -3), (2, 12))


@tvm.testing.uses_gpu
def test_dyn_shape_reshape():
    def verify_reshape(shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(newshape, "float32"))
        z = relay.reshape(x, relay.shape_of(y))

        func = relay.Function([x, y], z)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=newshape).astype("float32")
        ref_res = np.reshape(x_data, oshape)
        check_grad(run_infer_type(func), inputs=[x_data, y_data], eps=1e-3)
        verify_func(func, [x_data, y_data], ref_res)

    verify_reshape((2, 3, 4), (8, 3), (8, 3))
    verify_reshape((4, 7), (2, 7, 2), (2, 7, 2))


@tvm.testing.uses_gpu
def test_dyn_tile():
    def verify_tile(dshape, reps):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        r = relay.var("reps", relay.TensorType((len(reps),), "float32"))
        z = relay.tile(x, r)

        func = relay.Function([x, r], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.tile(x_data, reps=reps)
        reps_data = np.array(reps).astype("float32")
        verify_func(func, [x_data, np.array(reps).astype("float32")], ref_res)

    verify_tile((2, 3, 4), (3, 2, 1))
    verify_tile((2, 3, 4), (1, 2))
    verify_tile((2, 3), (3, 2, 1))


@tvm.testing.uses_gpu
def test_dyn_zeros_ones():
    def verify_zeros_ones(shape, dtype):
        for op, ref in [(relay.zeros, np.zeros), (relay.ones, np.ones)]:
            rank = len(shape)
            dyn_shape = relay.Var("shape", relay.ty.TensorType((rank,), "int64"))
            y = op(dyn_shape, dtype)
            yy = run_infer_type(y)
            assert yy.checked_type == relay.ty.TensorType((relay.Any(),) * rank, dtype)

            func = relay.Function([dyn_shape], y)
            ref_res = ref(shape, dtype)
            verify_func(func, [np.array(shape).astype("int64")], ref_res.astype("int64"))

    verify_zeros_ones((1, 3), "int64")
    verify_zeros_ones((8, 9, 1, 2), "float32")


@tvm.testing.uses_gpu
def test_dyn_full():
    def verify_full(fill_value, src_shape, dtype):
        x = relay.var("x", relay.scalar_type(dtype))
        rank = len(src_shape)
        dyn_src_shape = relay.var("dyn_scr_shape", relay.ty.TensorType((rank,), "int64"))
        z = relay.full(x, dyn_src_shape, dtype)
        func = relay.Function([x, dyn_src_shape], z)
        ref_res = np.full(src_shape, fill_value).astype(dtype)

        verify_func(
            func, [np.array(fill_value).astype(dtype), np.array(src_shape).astype("int64")], ref_res
        )

    verify_full(4, (1, 3, 4, 4), "int32")
    verify_full(4, (1, 3, 4, 4), "int64")
    verify_full(4.0, (2, 50), "float32")


@tvm.testing.uses_gpu
def test_dyn_sparse_to_dense():
    def verify_sparse_to_dense(sparse_indices, sparse_values, default_value, output_shape, xpected):
        sparse_indices_data = np.array(sparse_indices)
        sparse_values_data = np.array(sparse_values)
        default_value_data = np.array(default_value)
        output_shape_data = np.array(output_shape)

        a = relay.var(
            "a", relay.TensorType(sparse_indices_data.shape, str(sparse_indices_data.dtype))
        )
        b = relay.var(
            "b", relay.TensorType(sparse_values_data.shape, str(sparse_values_data.dtype))
        )
        output_shape_var = relay.var(
            "output_shape", relay.TensorType(output_shape_data.shape, str(output_shape_data.dtype))
        )
        if default_value is None:
            args = [a, b, output_shape_var]
            d = relay.sparse_to_dense(a, output_shape_var, b)
        else:
            c = relay.var(
                "c", relay.TensorType(default_value_data.shape, str(default_value_data.dtype))
            )
            args = [a, b, c, output_shape_var]
            d = relay.sparse_to_dense(a, output_shape_var, b, c)

        zz = run_infer_type(d)
        assert len(zz.checked_type.shape) == len(output_shape)

        func = relay.Function(args, d)

        if default_value is None:
            arguments = [sparse_indices_data, sparse_values_data, output_shape_data]
        else:
            arguments = [
                sparse_indices_data,
                sparse_values_data,
                default_value_data,
                output_shape_data,
            ]

        verify_func(func, arguments, xpected)

    verify_sparse_to_dense(1, 3, 0, [5], [0, 3, 0, 0, 0])  # scalar
    verify_sparse_to_dense([0, 1, 4], [3, 3, 3], 0, [5], [3, 3, 0, 0, 3])  # vector
    verify_sparse_to_dense(
        [[0, 0], [1, 2]], [1, 2], 0, [3, 4], [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]
    )  # nXd
    verify_sparse_to_dense(
        [[0, 0, 0], [1, 2, 3]],
        [1, 2],
        4,
        [2, 3, 4],
        [[[1, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]], [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 2]]],
    )  # nXd
    verify_sparse_to_dense(
        [0, 1, 4], [3.1, 3.1, 3.1], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1]
    )  # floats
    verify_sparse_to_dense(1, 3, None, [5], [0, 3, 0, 0, 0])  # default value not specified


if __name__ == "__main__":
    pytest.main([__file__])
