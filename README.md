# mnnl

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](https://isocpp.org/)
[![Vulkan](https://img.shields.io/badge/GPU-Vulkan-AC162C.svg)](https://www.vulkan.org/)
[![Build: Axle](https://img.shields.io/badge/build-Axle-orange.svg)](https://github.com/striter-no/axle)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Version](https://img.shields.io/badge/version-0.0.0-orange.svg)
![Status](https://img.shields.io/badge/status-in%20development-orange.svg)

**mnnl** is an experimental modular C++20 library for tensor operations, basic neural-network training, and GPU compute through Vulkan.

The project currently includes its own N-dimensional tensor type, dense layers, activation and loss interfaces, a simple sequential network, dataset helpers, and an early Vulkan compute backend built on top of [ygpu](https://github.com/striter-no/ygpu).

> [!IMPORTANT]
> mnnl is at a very early stage of development. The public API, tensor conventions, module layout, and GPU architecture are not stable yet. The project is suitable for learning, experimentation, and development of the library itself, but not for production workloads.

## Goals

mnnl is being developed as a small, understandable neural-network library where the complete path from tensor storage to GPU command submission remains visible in the source code.

The project focuses on:

- implementing the basic building blocks of neural networks without a large framework dependency;
- keeping tensor and training code readable and explicit;
- separating math, machine-learning, network, data, backend, and GPU responsibilities into Axle modules;
- supporting CPU execution first and gradually adding Vulkan compute acceleration;
- allowing direct access to lower-level GPU objects when a higher-level abstraction is not ready;
- keeping the codebase useful as an experimental platform for new backends and operators.

mnnl is not intended to replace mature frameworks such as PyTorch, TensorFlow, or ONNX Runtime.

## Current capabilities

### Tensor and math

- N-dimensional row-major tensors
- Runtime tensor shapes and strides
- Indexed access with bounds checking
- Reshaping without changing the total element count
- Scalar fill and vector-based construction
- Uniform and normal random initialization
- Element-wise transformation through callbacks
- Scalar addition and multiplication
- Element-wise tensor addition and Hadamard multiplication
- 2D transpose
- Matrix multiplication
- Broadcasting of batch dimensions for N-dimensional matrix multiplication
- Basic tensor string formatting
- Common numeric aliases such as `f32`, `f64`, `i32`, and `u32`

### Machine learning

- Dense `Linear` layer
- Forward and backward passes
- Weight and bias gradients
- He, Xavier, and uniform weight initialization
- Activation-function interface
- ReLU activation
- Loss-function interface
- Mean squared error
- Optimizer interface
- Stochastic gradient descent

### Networks and datasets

- Sequential `LinearNetwork`
- Optional activation function after each layer
- Forward-pass cache used during backpropagation
- Layer-by-layer backward propagation
- Weight updates through an optimizer
- In-memory datasets
- Random sample selection
- Batched dataset construction
- Optional batch shuffling

### Experimental Vulkan compute

- Vulkan device discovery and explicit device selection
- GPU, upload-staging, and readback-staging buffers
- Loading precompiled SPIR-V compute shaders
- Runtime GLSL compilation through ygpu
- SPIR-V reflection of compute workgroup sizes
- Storage, read-only storage, and uniform bindings
- Compute pipeline and descriptor construction
- Push constants
- Automatic dispatch-group calculation from shader local sizes
- Staging-to-GPU and GPU-to-staging copies
- Transfer/compute synchronization barriers
- Asynchronous queue submission through fences
- Explicit waiting for a submitted compute workload

### Utilities

- Loading binary files into byte arrays
- Saving byte arrays to binary files
- File-based tests managed by Axle

## Architecture

mnnl is split into small Axle modules instead of being built as one monolithic library.

| Module | Status | Responsibility |
|---|---|---|
| `math` | Experimental, usable | Tensor storage, shape handling, initialization, element-wise operations, transpose, and matrix multiplication |
| `ml` | Experimental | Layers, activations, losses, gradients, and optimizers |
| `networks` | Experimental | Sequential linear networks and in-memory datasets |
| `datahelper` | Early | Binary data loading and saving |
| `backend` | Scaffold | Future CPU/GPU backend interfaces and implementation separation |
| `gpu` | Prototype | Vulkan compute devices, buffers, shaders, layouts, and pipelines |
| `_` | Aggregate | Axle module that exposes the complete current project dependency graph |
| `main` | Example | Standalone GPU compute example |

The CPU-side math and training code currently operate directly on `nn::math::Tensor`. The Vulkan module is a separate compute layer and is not yet automatically selected as a tensor execution backend.

## Requirements

### Core development

- a C++20-capable compiler;
- [Axle](https://github.com/striter-no/axle);
- a Unix-like build environment supported by the current Axle setup.

The current project configuration uses `clang++`.

### GPU development

Building the complete aggregate module currently also requires:

- Vulkan headers;
- a Vulkan loader or implementation;
- a Vulkan-capable GPU and driver for hardware tests;
- [ygpu](https://github.com/striter-no/ygpu);
- SPIR-V Reflect;
- `glslc` when GLSL shaders are compiled at runtime.

The `gpu` Axle module depends on the exported modules `ygpu` and `spirv-reflect`.

> [!NOTE]
> CPU-only parts of the source do not conceptually require Vulkan. However, the current aggregate module includes the GPU module, so the complete default project build may still require the Vulkan toolchain.

## Building

Install Axle and run the build from the repository root:

```bash
axle build .
```

Force recompilation of local sources:

```bash
axle build . --clean
```

Update top-level remote dependencies before building:

```bash
axle build . --update
```

Because shader and asset paths are currently relative, examples and GPU tests should be started from the repository root.

Run the current example executable:

```bash
./.modules/main/bin/main
```

The example:

1. enumerates available Vulkan devices;
2. creates a device through ygpu;
3. creates upload, GPU-input, GPU-output, and readback buffers;
4. uploads a vector of floating-point values;
5. loads a compute shader;
6. submits the compute workload asynchronously;
7. copies the result back to host-visible memory.

## Quick start

The following example performs one training step on a small dense network:

```cpp
#include <iostream>
#include <memory>

#include <ml/actfuncs.hpp>
#include <ml/lossfs.hpp>
#include <ml/optimizer.hpp>
#include <netw/network.hpp>
#include <nnmath/tensor.hpp>

int main()
{
    using namespace nn;

    auto input =
        math::Tensor<f32>::create({0.0f, 1.0f});

    auto expected =
        math::Tensor<f32>::create({1.0f});

    LinearNetwork<f32> network;

    network.add_layer(
        layers::Linear<f32>(2, 4),
        std::make_shared<actf::ReLU<f32>>());

    network.add_layer(
        layers::Linear<f32>(4, 1));

    loss::MSE<f32> loss_function;
    optim::SGD<f32> optimizer(0.01f);

    auto predicted =
        network.forward(input);

    const auto loss_value =
        loss_function.forward(predicted, expected);

    auto gradient =
        loss_function.backward(predicted, expected);

    network.backward(gradient);
    network.update_weights(optimizer);

    std::cout
        << "Loss: " << loss_value << '\n'
        << "Output: " << predicted.to_string() << '\n';
}
```

The API is still evolving, so examples in `.tests/` should be treated as the most current usage reference.

## Tensor example

```cpp
#include <iostream>

#include <nnmath/tensor.hpp>

int main()
{
    using nn::math::Tensor;

    Tensor<float> a({2, 3});
    a.fill({
        {1.0f, 2.0f},
        {3.0f, 4.0f},
        {5.0f, 6.0f}
    });

    Tensor<float> b({3, 2});
    b.fill({
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    });

    auto result = a.matmul(b);

    std::cout << result.to_string() << '\n';
}
```

N-dimensional `matmul` supports broadcasting across batch dimensions. For example, multiplying tensors with shapes `[2, 1, 3, 4]` and `[1, 2, 4, 5]` produces a tensor with shape `[2, 2, 3, 5]`.

## Training a small network

The current network API is intentionally explicit:

```cpp
Dataset dataset;

dataset.add_entry(
    math::Tensor<f32>::create({0, 1}),
    math::Tensor<f32>::create({1}));

dataset.add_entry(
    math::Tensor<f32>::create({0, 0}),
    math::Tensor<f32>::create({0}));

dataset.add_entry(
    math::Tensor<f32>::create({1, 0}),
    math::Tensor<f32>::create({1}));

dataset.add_entry(
    math::Tensor<f32>::create({1, 1}),
    math::Tensor<f32>::create({0}));

LinearNetwork network;

network.add_layer(
    layers::Linear(2, 4),
    std::make_shared<actf::ReLU<f32>>());

network.add_layer(
    layers::Linear(4, 1));

loss::MSE loss_function;
optim::SGD optimizer(0.01f);

for (int epoch = 0; epoch < 100000; ++epoch) {
    auto batch = dataset.batch(0, true);
    auto prediction = network.forward(batch.input);

    auto gradient =
        loss_function.backward(
            prediction,
            batch.true_output);

    network.backward(gradient);
    network.update_weights(optimizer);
}
```

This example is primarily a regression and development test. mnnl does not yet provide a high-level trainer, callbacks, validation loops, checkpointing, or automatic differentiation.

## GPU compute model

The experimental GPU layer is currently based on explicit objects:

- `GPUDevice` owns a ygpu Vulkan device;
- `Buffer<T>` wraps GPU or staging memory;
- `ComputeShader` owns shader bytecode and reflected workgroup sizes;
- `ComputeLayout` describes buffer bindings and push constants;
- `ComputePipeline` builds descriptors and pipeline objects, records commands, submits work, and waits through a fence.

The intended flow is:

```text
create device
    ↓
create staging and GPU buffers
    ↓
fill upload buffer
    ↓
load SPIR-V or GLSL compute shader
    ↓
describe bindings and push constants
    ↓
build compute pipeline
    ↓
record copies, barriers, dispatch, and readback
    ↓
submit asynchronously
    ↓
wait and fetch the result
```

This API is lower-level than the CPU tensor API. Automatic operator selection, tensor-to-buffer synchronization, GPU memory planning, and graph execution are future work.

## Testing

Build the main project before running the tests:

```bash
axle build .
axle test
```

Run one test by filename without the `.cpp` extension:

```bash
axle test tensors
axle test backward
axle test network
axle test raw_vulkan
```

Build tests without executing them:

```bash
axle test --only-build
```

The current test and development programs cover:

- basic executable setup;
- tensor indexing, formatting, random initialization, and transformations;
- N-dimensional broadcasting matrix multiplication;
- manual layer forward and backward propagation;
- training a small XOR-style dataset;
- direct Vulkan compute through ygpu;
- GPU buffer transfers, descriptors, barriers, dispatch, and readback.

`raw_vulkan` requires a usable Vulkan implementation and the corresponding shader assets.

## Project structure

```text
.
├── .modules/
│   ├── _/                  # aggregate Axle module
│   ├── backend/            # future execution backends
│   ├── datahelper/         # data and binary-file helpers
│   ├── gpu/                # experimental Vulkan compute layer
│   ├── main/               # current example executable
│   ├── math/               # Tensor and numeric utilities
│   ├── ml/                 # layers, activations, losses, optimizers
│   └── networks/           # networks and datasets
├── .tests/
│   ├── backward.cpp
│   ├── default.cpp
│   ├── network.cpp
│   ├── raw_vulkan.cpp
│   ├── tensors.cpp
│   ├── defaults.json
│   └── module.json
├── assets/                 # compute shaders and other example assets
├── defaults.json           # shared Axle compiler targets
├── module.json             # root Axle project
└── LICENSE
```

The exact root layout may change while the build configuration and public modules are being finalized.

## Roadmap

Likely development directions include:

- stabilizing tensor shape and indexing conventions;
- adding tensor subtraction, division, reductions, slicing, and broader broadcasting;
- separating storage from execution backends;
- introducing a CPU backend behind a common operator interface;
- connecting tensor operators to the Vulkan backend;
- adding GPU tensor allocation and synchronization;
- implementing more layers, activations, losses, and optimizers;
- replacing manual backward implementations with a more general gradient system;
- adding model and optimizer serialization;
- improving dataset loading and batching;
- adding deterministic random seeds;
- expanding numerical and gradient tests;
- improving Vulkan resource lifetime and error handling;
- supporting reusable command submission and multiple in-flight workloads;
- documenting the public API as it stabilizes.

This list describes possible directions rather than a compatibility promise.

## Known limitations

- mnnl is pre-1.0 and has no stable public API.
- Only a small subset of tensor operations is implemented.
- Training currently focuses on dense linear networks.
- There is no general automatic-differentiation graph.
- There are no convolutional, recurrent, normalization, dropout, or attention layers.
- MSE and SGD are currently the primary implemented loss and optimizer.
- The existing `Sigmoid` implementation is unfinished and should not yet be treated as correct or stable.
- Tensor storage is host-side and is not transparently connected to the Vulkan backend.
- The `backend` module is currently scaffolding rather than a complete backend abstraction.
- GPU compute requires manually managed buffers, bindings, synchronization, and shaders.
- GPU shader and asset paths are currently relative to the repository working directory.
- GPU tests require suitable Vulkan hardware or a compatible software Vulkan implementation.
- Error handling is still inconsistent between exceptions, returned error values, and raw Vulkan results.
- Model serialization and a stable checkpoint format are not implemented.
- Dataset objects currently keep all entries in memory.
- Numerical correctness, gradient correctness, and edge-case coverage are still limited.
- Build and development workflows currently assume Axle and a Unix-like environment.

## Contributing

Issues, bug reports, tests, documentation improvements, and pull requests are welcome.

Useful contribution areas include:

- tensor correctness and shape validation;
- numerical and finite-difference gradient tests;
- CPU operator implementations;
- backend-interface design;
- Vulkan synchronization and resource ownership;
- additional compute shaders;
- serialization;
- dataset loaders;
- API documentation;
- portable build configurations.

Because the architecture is still evolving, discussing large API or backend changes before implementation is recommended.

## License

mnnl is free and open-source software licensed under the **GNU General Public License v3.0**.

See [`LICENSE`](LICENSE) for the full license text.

Third-party dependencies retain their respective licenses.
