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

"""Unit tests for the CanonicalizeShapeExpr pass"""

import pytest
import tvm
import tvm.testing
from tvm import relax
from tvm.script import relax as R
from tvm.script import tir as T


def test_simple_compound_shape():
    """Test canonicalization of simple compound shape expression"""

    @R.function
    def before(x: R.Tensor(("n",), "float32")):
        n = T.int64()
        # Compound expression: n + 1
        y: R.Tensor((n + 1,), "float32") = R.zeros(R.shape([n + 1]), dtype="float32")
        return y

    mod = tvm.IRModule.from_expr(before)
    mod = relax.transform.CanonicalizeShapeExpr()(mod)
    mod = relax.transform.ComputePrimValue()(mod)
    mod = relax.transform.VMShapeLower()(mod)

    assert "compute_symbolic_expr" in [str(gv) for gv in mod.get_global_vars()]


def test_compound_shape_in_constant():
    """Test canonicalization when compound shape appears in constant variable struct_info"""

    @R.function
    def before(x: R.Tensor(("n", "m"), "float32")):
        n = T.int64()
        m = T.int64()
        # This pattern can occur after FoldConstant inlines shapes
        # The constant variable has compound expression in its struct_info
        y: R.Tensor((n * m,), "float32") = R.zeros(R.shape([n * m]), dtype="float32")
        return y

    mod = tvm.IRModule.from_expr(before)
    print("=== Before CanonicalizeShapeExpr ===")
    print(mod)
    mod = relax.transform.CanonicalizeShapeExpr()(mod)
    print("=== After CanonicalizeShapeExpr ===")
    print(mod)

    # Check well-formed immediately after CanonicalizeShapeExpr
    is_wf = relax.analysis.well_formed(mod)
    print(f"=== Well-formed after CanonicalizeShapeExpr: {is_wf} ===")
    if not is_wf:
        raise RuntimeError("IR is not well-formed after CanonicalizeShapeExpr")

    mod = relax.transform.Normalize()(mod)
    print("=== After Normalize ===")
    print(mod)

    # Check well-formed immediately after Normalize
    is_wf = relax.analysis.well_formed(mod)
    print(f"=== Well-formed after Normalize: {is_wf} ===")
    if not is_wf:
        raise RuntimeError("IR is not well-formed after Normalize")

    mod = relax.transform.ComputePrimValue()(mod)
    print("=== After ComputePrimValue ===")
    print(mod)

    # Check well-formed immediately after ComputePrimValue
    is_wf = relax.analysis.well_formed(mod)
    print(f"=== Well-formed after ComputePrimValue: {is_wf} ===")
    if not is_wf:
        raise RuntimeError("IR is not well-formed after ComputePrimValue")

    mod = relax.transform.VMShapeLower()(mod)
    print("=== After VMShapeLower ===")
    print(mod)

    # Verify a compute function was generated for the compound expression
    assert "compute_symbolic_expr" in [str(gv) for gv in mod.get_global_vars()]


def test_multiply_compound_shape():
    """Test the original issue case: 4 * x_0 * x_1 * x_2 * x_3"""

    @R.function
    def before(x: R.Tensor(("n", "m", "p", "q"), "float32")):
        n = T.int64()
        m = T.int64()
        p = T.int64()
        q = T.int64()
        # Compound expression: 4 * n * m * p * q
        y: R.Tensor((4 * n * m * p * q,), "float32") = R.zeros(
            R.shape([4 * n * m * p * q]), dtype="float32"
        )
        return y

    mod = tvm.IRModule.from_expr(before)
    mod = relax.transform.CanonicalizeShapeExpr()(mod)
    mod = relax.transform.ComputePrimValue()(mod)
    mod = relax.transform.VMShapeLower()(mod)

    # Verify a compute function was generated for the compound expression
    assert "compute_symbolic_expr" in [str(gv) for gv in mod.get_global_vars()]


def test_no_change_for_canonical_shape():
    """Test that already canonical shapes are not modified"""

    @R.function
    def before(x: R.Tensor(("n",), "float32")):
        n = T.int64()
        # Already canonical shape
        y: R.Tensor((n,), "float32") = R.zeros(R.shape([n]), dtype="float32")
        return y

    mod_before = tvm.IRModule.from_expr(before)
    mod_after = relax.transform.CanonicalizeShapeExpr()(mod_before)

    # The mod should be unchanged (or minimally changed)
    # Both should work with VMShapeLower
    mod_before_lower = relax.transform.ComputePrimValue()(mod_before)
    mod_before_lower = relax.transform.VMShapeLower()(mod_before_lower)
    mod_after_lower = relax.transform.ComputePrimValue()(mod_after)
    mod_after_lower = relax.transform.VMShapeLower()(mod_after_lower)

    # For canonical shapes, no compute_symbolic_expr should be generated
    # (only compound expression need computation)
    global_var_names = [str(gv) for gv in mod_after_lower.get_global_vars()]
    assert "compute_symbolic_expr" not in global_var_names


def test_no_change_for_concrete_shape():
    """Test that concrete integer shapes are not modified"""

    @R.function
    def before(x: R.Tensor((10,), "float32")):
        # Concrete shape
        y: R.Tensor((10,), "float32") = R.zeros(R.shape([10]), dtype="float32")
        return y

    mod = tvm.IRModule.from_expr(before)
    mod = relax.transform.CanonicalizeShapeExpr()(mod)
    mod = relax.transform.ComputePrimValue()(mod)
    mod = relax.transform.VMShapeLower()(mod)

    # For concrete shapes, no compute_symbolic_expr should be generated
    global_var_names = [str(gv) for gv in mod.get_global_vars()]
    assert "compute_symbolic_expr" not in global_var_names


def test_tuple_struct_info():
    """Test canonicalization with tuple struct info containing compound shapes"""

    @R.function
    def before(x: R.Tensor(("n",), "float32")):
        n = T.int64()
        # Tuple with compound shapes
        y: R.Tuple(R.Tensor((n + 1,), "float32"), R.Tensor((n * 2,), "float32")) = (
            R.zeros(R.shape([n + 1]), dtype="float32"),
            R.zeros(R.shape([n * 2]), dtype="float32"),
        )
        return y

    mod = tvm.IRModule.from_expr(before)
    mod = relax.transform.CanonicalizeShapeExpr()(mod)
    mod = relax.transform.ComputePrimValue()(mod)
    mod = relax.transform.VMShapeLower()(mod)

    # Verify compute functions were generated for the compound expressions
    assert "compute_symbolic_expr" in [str(gv) for gv in mod.get_global_vars()]


def test_full_pipeline_with_opt_level_1():
    """Test the full pipeline with opt_level=1"""

    @R.function
    def before(x: R.Tensor(("n", "m"), "float32")):
        n = T.int64()
        m = T.int64()
        y: R.Tensor((n * m,), "float32") = R.reshape(x, R.shape([n * m]))
        return y

    mod = tvm.IRModule.from_expr(before)

    with tvm.transform.PassContext(opt_level=1):
        # Apply the passes in order
        mod = relax.transform.LegalizeOps()(mod)
        mod = relax.transform.AnnotateTIROpPattern()(mod)
        mod = relax.transform.FoldConstant()(mod)
        mod = relax.transform.ComputePrimValue()(mod)
        mod = relax.transform.CanonicalizeShapeExpr()(mod)
        mod = relax.transform.VMShapeLower()(mod)

    # Verify a compute function was generated for the compound expression
    assert "compute_symbolic_expr" in [str(gv) for gv in mod.get_global_vars()]


if __name__ == "__main__":
    import sys

    print("Running CanonicalizeShapeExpr unit tests...")
    print("=" * 80)

    tests = [
        ("Simple compound shape", test_simple_compound_shape),
        ("Compound shape in constant", test_compound_shape_in_constant),
        ("Multiply compound shape", test_multiply_compound_shape),
        ("No change for canonical shape", test_no_change_for_canonical_shape),
        ("No change for concrete shape", test_no_change_for_concrete_shape),
        ("Tuple struct info", test_tuple_struct_info),
        ("Full pipeline with opt_level=1", test_full_pipeline_with_opt_level_1),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"\nTest: {name}")
            test_func()
            print("Result: PASSED")
            passed += 1
        except Exception as e:
            print(f"Result: FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"Total tests run: {passed + failed}, Passed: {passed}, Failed: {failed}")

    sys.exit(0 if failed == 0 else 1)
