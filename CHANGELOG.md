# Changelog

All notable changes to RoLLM are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0] - 2025-05-17

### Added

- `RoLLM.new` — single entry-point constructor accepting corpus text and `TransformerConfig`.
- `CharTokenizer` — character-level tokenizer with full vocabulary build from training text.
- `BPETokenizer` — Byte-Pair Encoding tokenizer with external JSON vocab support via `HttpService`.
- `Embedding` — learnable token embeddings plus pre-computed sinusoidal positional encodings.
- `MultiHeadAttention` — scaled dot-product attention with causal mask, full forward and backward.
- `FeedForward` — two-layer ReLU feed-forward network with bias, forward and backward.
- `LayerNorm` — per-row layer normalisation with learnable gamma/beta, forward and backward.
- `TransformerBlock` — residual MHA + norm + FFN + norm block.
- `TransformerModel` — N-layer transformer with causal mask cache and position cache.
- `CrossEntropyLoss` — numerically stable softmax cross-entropy, forward and backward.
- `Optimizer` — Adam optimiser with bias correction.
- `trainModel` — full training loop returning per-epoch history (`epoch`, `loss`, `averageLoss`, `samples`).
- `trainModel` `TrainingOptions` — `yieldEverySamples` option to yield between samples and prevent script timeout.
- `generate` / `generateTemperature` — greedy and temperature-sampled autoregressive generation.
- `predict` / `predictTemperature` — single next-token prediction.
- `getParameterCount` — total trainable scalar parameter count.
- Examples: `TinyTraining.server.luau`, `InGameChatBot.server.luau`.
- Smoke test suite: `tests/SmokeTests.server.luau`.

### Performance

- Causal mask and position index arrays cached by sequence length.
- Training samples tokenized once before the epoch loop.
- `LinearAlgebra:mm` pre-transposes the right-hand matrix for row-major inner loop access.
- Attention softmax backward reuses cached softmax weights instead of recomputing `math.exp`.

[0.1.0]: https://github.com/rustyspottedcatt/RoLLM/releases/tag/v0.1.0
