// RUN: torch-mlir-opt -split-input-file -allow-unregistered-dialect %s -torch-raise-if-op-shapes-dtypes -mlir-disable-threading -debug-only=greedy-rewriter | FileCheck %s

// CHECK-LABEL:   func.func @control_flow$raise_shapes(
// CHECK-SAME:                                          %[[ARG0:.*]]: !torch.vtensor<[1,2,3],f32>, %[[COND:.*]]: !torch.bool) -> !torch.vtensor {
// CHECK:           %[[IFRES:.*]] = torch.prim.If %[[COND]] -> (!torch.vtensor<[1,2,3],f32>) {
// CHECK:             torch.prim.If.yield %[[ARG0]] : !torch.vtensor<[1,2,3],f32>
// CHECK:           } else {
// CHECK:             torch.prim.If.yield %[[ARG0]] : !torch.vtensor<[1,2,3],f32>
// CHECK:           }
// CHECK:           %[[CASTED:.*]] = torch.tensor_static_info_cast %[[IFRES]] : !torch.vtensor<[1,2,3],f32> to !torch.vtensor 
// CHECK:           return %[[CASTED:.*]] : !torch.vtensor
func.func @control_flow$raise_shapes(%arg0: !torch.vtensor<[1,2,3],f32>, %cond: !torch.bool) -> (!torch.vtensor) {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[1,2,3],f32> to !torch.vtensor
  %2 = torch.prim.If %cond -> (!torch.vtensor) {
    %1 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[1,2,3],f32> to !torch.vtensor
    torch.prim.If.yield %1 : !torch.vtensor
  } else {
    torch.prim.If.yield %0 : !torch.vtensor
  }
  return %2 : !torch.vtensor
}

// CHECK-LABEL:   func.func @control_flow$nested(
// CHECK-SAME:                                          %[[ARG0:.*]]: !torch.vtensor<[1,2,3],f32>, %[[COND:.*]]: !torch.bool) -> !torch.vtensor {
// CHECK:           %[[IFRES:.*]] = torch.prim.If %[[COND]] -> (!torch.vtensor<[1,2,3],f32>) {
// CHECK:             %[[IFNESTED:.*]] = torch.prim.If %[[COND]] -> (!torch.vtensor<[1,2,3],f32>) {
// CHECK:               torch.prim.If.yield %[[ARG0]] : !torch.vtensor<[1,2,3],f32>
// CHECK:             } else {
// CHECK:               torch.prim.If.yield %[[ARG0]] : !torch.vtensor<[1,2,3],f32>
// CHECK:             }
// CHECK:             torch.prim.If.yield %[[IFNESTED]] : !torch.vtensor<[1,2,3],f32> 
// CHECK:           } else {
// CHECK:             torch.prim.If.yield %[[ARG0]] : !torch.vtensor<[1,2,3],f32>
// CHECK:           }
// CHECK:           %[[CASTED:.*]] = torch.tensor_static_info_cast %[[IFRES]] : !torch.vtensor<[1,2,3],f32> to !torch.vtensor 
// CHECK:           return %[[CASTED:.*]] : !torch.vtensor
func.func @control_flow$nested(%arg0: !torch.vtensor<[1,2,3],f32>, %cond: !torch.bool) -> (!torch.vtensor) {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[1,2,3],f32> to !torch.vtensor
  %1 = torch.prim.If %cond -> (!torch.vtensor) {
    %2 = torch.prim.If %cond -> (!torch.vtensor) {
      %3 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[1,2,3],f32> to !torch.vtensor
      torch.prim.If.yield %3 : !torch.vtensor
    } else {
      torch.prim.If.yield %0 : !torch.vtensor
    }
    torch.prim.If.yield %2 : !torch.vtensor
  } else {
    %3 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[1,2,3],f32> to !torch.vtensor
    torch.prim.If.yield %3 : !torch.vtensor
  }

  return %1 : !torch.vtensor
}
