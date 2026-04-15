/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file src/relax/backend/vm/vm_shape_lower.cc
 * \brief Lower the function boundary type checks and symbolic shape computations.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tirx/analysis.h>

namespace tvm {
namespace relax {
namespace transform {

/*!
 * \brief Canonicalize ShapeExpr by extracting composite PrimExpr into separate bindings.
 *
 * Before:
 *   lv = relax.call_tir(func, (x,), shape=[4 * n + 1])
 *
 * After:
 *   s1 = R.prim_value(4 * n + 1)
 *   lv = relax.call_tir(func, (x,), shape=[s1])
 */
class ShapeExprCanonicalizer : public ExprMutator {
 public:
  static IRModule Canonicalize(IRModule mod) {
    ShapeExprCanonicalizer mutator;
    IRModule new_mod = mod;

    for (auto& kv : mod->functions) {
      if (auto* func = kv.second.as<FunctionNode>()) {
        Function updated_func = 
            Downcast<Function>(mutator.VisitExpr(GetRef<Function>(kv.second)));
        new_mod = new_mod->Update(kv.first, updated_func);
      }
    }
    return new_mod;
  }

 private:
  // Check if a PrimExpr is "composite" (requires computation)
  static bool IsCompositeExpr(const PrimExpr& expr) {
    // Simple expressions: IntImm or Var
    if (expr.as<IntImmNode>()) return false;
    if (expr.as<tirx::VarNode>()) return false;
    
    // Everything else is composite and needs extraction
    return true;
  }

  // Extract composite PrimExpr into a PrimValue binding
  Expr ExtractCompositeShape(const PrimExpr& expr) {
    // Create a PrimValue expression for this shape computation
    PrimValue prim_val(expr);
    
    // Create a new variable to hold the computed shape
    PrimStructInfo prim_sinfo(DataType::Int(64));
    Var shape_var("s", prim_sinfo);
    
    // Emit the binding: shape_var = prim_value(expr)
    builder_->Emit(prim_val, "s");
    
    // Return the variable reference
    return shape_var;
  }

  // Override VisitExpr_ for ShapeExpr
  Expr VisitExpr_(const ShapeExprNode* op) final {
    // Check if all values are simple (no extraction needed)
    bool all_simple = std::all_of(
        op->values.begin(), op->values.end(),
        [](const PrimExpr& e) { return !IsCompositeExpr(e); });
    
    if (all_simple) {
      // No composite expressions, return as-is
      return GetRef<Expr>(op);
    }

    // Extract composite expressions
    Array<PrimExpr> new_values;
    for (const PrimExpr& expr : op->values) {
      if (IsCompositeExpr(expr)) {
        // Extract into a binding and use the variable
        Expr extracted = ExtractCompositeShape(expr);
        if (auto* var = extracted.as<VarNode>()) {
          new_values.push_back(Variable(var->vid));
        }
      } else {
        // Keep simple expressions as-is
        new_values.push_back(expr);
      }
    }

    return ShapeExpr(new_values);
  }

  // Override VisitExpr_ for PrimValue (may also contain composite exprs)
  Expr VisitExpr_(const PrimValueNode* op) final {
    if (IsCompositeExpr(op->value)) {
      return ExtractCompositeShape(op->value);
    }
    return GetRef<Expr>(op);
  }

  // Override VisitBinding_ to handle bindings
  void VisitBinding_(const VarBindingNode* binding) final {
    Expr value = this->VisitExpr(binding->value);
    
    // Emit the original binding with visited value
    builder_->Emit(VarBinding(binding->var, value), binding->var->name_hint());
  }
};

Pass ShapeExprCanonicalize() {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    return ShapeExprCanonicalizer::Canonicalize(mod);
  };

  return CreateModulePass(pass_func, 0, "ShapeExprCanonicalize", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.ShapeExprCanonicalize", []() {
    return ShapeExprCanonicalize();
  });
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm