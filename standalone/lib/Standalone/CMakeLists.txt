add_mlir_dialect_library(MLIRStandalone
        StandaloneTypes.cpp
        StandaloneDialect.cpp
        StandaloneOps.cpp
        StandalonePasses.cpp
        StandaloneTilePasses.cpp
        StandaloneEpilogueFusion.cpp
        StandaloneTilingInterfaces.cpp
        StandaloneTransformOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        MLIRStandaloneOpsIncGen
        MLIRStandalonePassesIncGen
        MLIRStandaloneTransformOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRMemRefDialect
        MLIRSCFDialect
        MLIRSCFTransforms
        MLIRTilingInterface
        MLIRDestinationStyleOpInterface
        MLIRTensorDialect
        MLIRTransformDialect
        MLIRTransformDialectInterfaces
        MLIRTransformDialectUtils
        )
