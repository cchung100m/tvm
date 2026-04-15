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

import tvm
from tvm import relax
import numpy as np

def test_composite_shape():
    """Test that composite shapes are properly canonicalized."""
    
    # Define a simple Relax function with composite shape expression
    @relax.function
    def main(x: relax.Tensor(["n"], "float32")) -> relax.Tensor(["n * 4"], "float32"):
        # Composite shape: n * 4
        # This previously caused: "PrimExpr has not been computed"
        lv0 = relax.op.reshape(x, relax.shape([relax.const(4) * "n"]))
        return lv0
    
    mod = tvm.IRModule({"main": main})
    
    print("Original module:")
    print(mod)
    
    # Run the canonicalization pass
    print("\n" + "="*60)
    print("Running ShapeExprCanonicalize...")
    canonicalized = relax.transform.ShapeExprCanonicalize()(mod)
    
    print("\nCanonical module:")
    print(canonicalized)
    
    print("\n" + "="*60)
    print("Running full pipeline with opt_level=1...")
    
    try:
        pipeline = relax.get_pipeline("relax.backend.cpu_generic", opt_level=1)
        lowered = pipeline(mod)
        print("SUCCESS: Pipeline completed without errors!")
        print("\nLowered module:")
        print(lowered)
    except Exception as e:
        print(f"FAILED: {str(e)[:500]}")

if __name__ == "__main__":
    test_composite_shape()