FROM ubuntu:24.04 AS llvm-build

ARG LLVM_COMMIT=8a688983ab0b5aa3ac3cfbcd94d3727d023a169e

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    cmake \
    git \
    ninja-build \
    python3 \
    zlib1g-dev \
    libzstd-dev \
    libxml2-dev \
  && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/llvm/llvm-project.git /src/llvm-project \
  && cd /src/llvm-project \
  && git checkout "$LLVM_COMMIT"

RUN cmake -S /src/llvm-project/llvm -B /build/llvm -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_PARALLEL_COMPILE_JOBS=2 \
    -DLLVM_PARALLEL_LINK_JOBS=1 \
    -DCMAKE_INSTALL_PREFIX=/opt/llvm

RUN cmake --build /build/llvm --target install --parallel 2

FROM ubuntu:24.04 AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    make \
    ninja-build \
    python3 \
    python3-dev \
    python3-venv \
    verilator \
    zlib1g \
    zlib1g-dev \
    libzstd1 \
    libzstd-dev \
    libxml2 \
    libxml2-dev \
  && rm -rf /var/lib/apt/lists/*

COPY --from=llvm-build /opt/llvm /opt/llvm
COPY requirements.txt /tmp/requirements.txt

RUN python3 -m venv /opt/venv \
  && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

ENV PATH="/opt/venv/bin:/opt/llvm/bin:${PATH}"
ENV LLVM_DIR="/opt/llvm/lib/cmake/llvm"
ENV MLIR_DIR="/opt/llvm/lib/cmake/mlir"
ENV LD_LIBRARY_PATH="/opt/llvm/lib"

WORKDIR /work
