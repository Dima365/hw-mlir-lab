add_mlir_dialect(StandaloneOps standalone)
add_mlir_doc(StandaloneDialect StandaloneDialect Standalone/ -gen-dialect-doc)
add_mlir_doc(StandaloneOps StandaloneOps Standalone/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS StandalonePasses.td)
mlir_tablegen(StandalonePasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRStandalonePassesIncGen)

set(LLVM_TARGET_DEFINITIONS StandaloneTransformOps.td)
mlir_tablegen(StandaloneTransformOps.h.inc --gen-op-decls)
mlir_tablegen(StandaloneTransformOps.cpp.inc --gen-op-defs)
add_public_tablegen_target(MLIRStandaloneTransformOpsIncGen)
