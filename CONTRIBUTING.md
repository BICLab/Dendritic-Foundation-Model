# Contributing

Thank you for helping improve the Dendritic Foundation Model research release.

## Before opening a change

1. Search existing issues and pull requests.
2. For substantial changes, open an issue describing the affected experiment, expected behavior, environment, and validation plan.
3. Keep changes focused on one task or integration.
4. Do not commit datasets, model weights, generated outputs, credentials, private cluster paths, or artifacts that you are not authorized to redistribute.

## Development environments

The repository contains multiple independent research stacks. Create a separate environment for the component you modify and follow its README:

- [`figure5/`](figure5/) for modality-specific benchmarks;
- [`figure6/model_comparison/`](figure6/model_comparison/) for model definitions/configurations;
- [`figure6/long_context/`](figure6/long_context/) for framework integration modules.

Include the Python, PyTorch, CUDA, GPU, and key dependency versions in bug reports. For distributed failures, also report world size, tensor/context parallel sizes, and the launch command with private paths removed.

## Pull-request checklist

- Explain the motivation and the experiment or figure affected.
- Preserve scientific settings unless the change explicitly documents a corrected or alternative protocol.
- Add a minimal validation command and report its result.
- Keep machine-specific paths configurable or clearly documented.
- Verify that every command names an existing entry file.
- Verify Markdown relative links and close all fenced code blocks.
- Update the nearest README when dependencies, data layout, configuration fields, or integration steps change.
- Retain copyright, attribution, and license notices in third-party or adapted code.

## Documentation style

Use concise technical English. Distinguish among:

- settings reported in the paper;
- defaults present in the released source;
- examples introduced only for illustration;
- missing artifacts or external-framework requirements.

Do not claim that example configurations are trained checkpoints, that the long-context modules form a complete training system, or that unpublished bibliographic fields are known.

## Third-party code

Changes derived from another project must identify the source and compatible license. Do not remove upstream notices. If licensing is unclear, do not submit the copied code; open an issue with a link to the source instead.

## License

Unless explicitly stated otherwise, contributions to the authors’ original
code and documentation are accepted under the
[Apache License 2.0](LICENSE). By submitting a contribution, you represent
that you have the right to provide it. Third-party files remain subject to
their upstream terms; see [`THIRD_PARTY.md`](THIRD_PARTY.md).

For security-sensitive reports, use the repository owner’s private reporting channel rather than a public issue when one is available.
