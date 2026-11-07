# IANE development guidelines

## Design workflow

- Design public APIs from the caller's perspective. Write a small example and document its expected behavior before implementing a new interface. Keep design sketches outside the build until their dependencies are ready and label them clearly.
- Keep image ownership separate from non-owning views. Pass views to algorithms and derive dimensionality from their types when the operation is dimension-independent. Support strided views and subviews where the algorithm permits.
- Use C++20 as the baseline for host and CUDA code. Verify toolchain support before using newer features or standard library facilities in device code.

## Execution and transfers

- Provide a CPU implementation for each built-in computation. Keep CPU-only use independent of GPU toolchains and runtimes. Use the same call shape and documented semantics across supported backends. Backend-specific extension points may have their own requirements.
- Infer the computation backend at compile time from the participating view types, including views nested in expressions. Reject incompatible combinations at compile time by default. Use an explicit policy when the caller needs to choose how to handle a mixed-backend operation.
- Use `iane::copy` both to materialize a readable expression into a compatible destination and to transfer data between supported memory kinds. Keep cross-backend computation and transfer separate by default. Favor a single pass for pointwise expressions when possible.
- Treat `iane::run_on` as a public utility. Share one callable across CPU and CUDA. For host input and output, use the host views directly on CPU and manage device allocation, upload and download around the callable on CUDA. The host output is ready on return. Let later examples guide extensions to this interface.
- Prioritize CPU-to-Vulkan and CUDA-to-Vulkan transfers for renderer integration. Document resource ownership and when the destination is ready for the renderer. Add Vulkan compute operations when a use case calls for them.

## Examples

- Make each example a standalone CLI utility with `--help`. Take input and output paths, algorithm parameters and backend selection from arguments or input files. Use 2D images for the first examples where appropriate. Projection and volume rendering retain a 3D input and 2D output.
- Support a JSON configuration file through the shared example CLI. Use long option names as keys and let explicit command-line arguments override file values.
- Put common backend dispatch, transfers and measurement in shared code. Time computation through completion on each backend. Log setup and transfers separately when applicable. Complete warm-up runs before sampling and exclude them from the reported statistics. Document which phases are timed.
- Accompany each example with an English README that describes the command, inputs, outputs, expected behavior and timing fields. State its current build status.

## C++ style and documentation

- Follow the formatting of the surrounding code. Use tabs for indentation and follow existing naming conventions, including snake_case for public functions and types. Keep changes focused and avoid unrelated reformatting.
- Prefix type template parameters with `T` and non-type template parameters with `t`, for example `typename TMyType` and `int tMyValue`.
- Prefix named function and lambda parameters with `a`, for example `aSource`. Prefix class instance members and nonpublic struct instance members with `m`, for example `mSize`. Public struct fields may keep unprefixed names.
- Favor const-correct views and explicit ownership. Use C++20 concepts for public template requirements when they improve diagnostics. Reuse existing operations and types where they fit.
- Keep device code compatible with the supported CUDA compiler. Use existing host and device annotation macros where needed. Avoid unnecessary allocations, data transfers and synchronization in algorithm code. Document these costs when they are part of an interface.
- Write code documentation, comments and example documentation in English. Use Doxygen comments for public types, functions and concepts. Describe the contract and use tags such as `@param`, `@tparam`, `@return` and `@throws` when relevant.
- Document constraints callers need to use an API correctly. Include ownership and view lifetime, supported layouts and backends, aliasing, boundary behavior, numerical guarantees and asynchronous completion where applicable. Use implementation comments to explain decisions that are hard to infer from the code.
- Add a short English comment above internal helper types explaining their role in the implementation. Use ordinary comments for these helpers.
