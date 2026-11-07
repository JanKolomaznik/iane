# iane
Image Algebra for N-dimensional Evaluation

## API documentation

Generate the HTML reference with Doxygen 1.15 or newer and Graphviz (`dot` on PATH).

```sh
(cd docs && doxygen Doxyfile)
```

Open `docs/documentation/html/index.html`. The **Topics** section includes the
IANE concepts and their requirements. The reference also includes SVG
graphs for classes, header dependencies and directories.

## Guides

- [Procedural views](docs/procedural_views.md)
- [CUDA texture images](docs/cuda_texture_images.md)
- [Prepared algorithms and shared scratch](docs/prepared_algorithms.md)

## Deconvolution examples

- [Wiener filtering](examples/wiener/README.md)
- [Richardson–Lucy deconvolution](examples/rl_deconvolution/README.md)
