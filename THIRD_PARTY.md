# Third-party software

This research release includes code adapted from or designed to integrate
with third-party projects. Those files remain subject to their upstream
licenses; the repository-level Apache License 2.0 does not override these
terms.

## License texts

The [`third_party_licenses/`](third_party_licenses/) directory preserves full
upstream license texts when a notice is not already bundled next to the
adapted code.

## Bundled or adapted components

- **Zoology** (`figure5/mqar/zoology/`): Apache License 2.0. The upstream
  notice is retained in `figure5/mqar/zoology/LICENSE.md`.
- **Mamba** (`figure5/static_image/ops/mamba/`): Apache License 2.0. The
  upstream notice is retained in `figure5/static_image/ops/mamba/LICENSE`.
- **minLSTM** (`figure5/static_image/ops/minLSTM/`): MIT License. The
  upstream notice is retained in `figure5/static_image/ops/minLSTM/LICENSE`.
- **Efficient spiking networks**: the SHD and SSC conversion scripts are
  adapted from [byin-cwi/Efficient-spiking-networks](https://github.com/byin-cwi/Efficient-spiking-networks).
  Its MIT notice is retained in
  `third_party_licenses/efficient-spiking-networks-MIT.txt`.
- **Flash Linear Attention (FLA)**: several linear-attention operators and
  Triton kernels derive from or require
  [sustcsonglin/flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention).
  Its MIT notice is retained in
  `third_party_licenses/flash-linear-attention-MIT.txt`.
- **Masked Autoencoders / Meta vision utilities**: files carrying Meta
  copyright headers in `figure5/neuromorphic_vision/` derive from
  [facebookresearch/mae](https://github.com/facebookresearch/mae).
  These files are covered by **CC BY-NC 4.0**, not a permissive software
  license. The full notice is retained in
  `third_party_licenses/mae-CC-BY-NC-4.0.txt`.
- **vLLM**: the SparseLA-H inference adapter targets vLLM v0.9.1 and
  includes a cache adapter derived from vLLM. Its Apache License 2.0 notice
  is retained in `third_party_licenses/vllm-Apache-2.0.txt`.
- **Hugging Face Transformers, Triton, causal-conv1d, and OpenMMLab**:
  model files and experiment adapters include upstream-derived code or
  require these projects at runtime. Preserve their file-level notices and
  comply with the versions and licenses documented by each upstream project.

## License interaction

The project-level Apache License 2.0 does not relicense third-party
components. In particular, distributions containing the listed Meta vision
utilities remain subject to the non-commercial terms of CC BY-NC 4.0.
