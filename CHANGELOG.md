# Changelog

All notable changes to RoLLM are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.2.0] - 2026-05-24

### Added

- **GELU activation** — `FeedForward` replaces ReLU with the exact `tanh`-based GELU. The backward pass uses the analytical derivative rather than a finite-difference approximation, so gradients are correct and cheap.
- **Xavier initialization** — `LinearAlgebra:xavierMatrix(rows, cols)` draws from a normal distribution with std = √(2 / (rows + cols)). Used for `Wq`, `Wk`, `Wv`, `Wo`, `W1`, and `W2` in place of the old small-constant random init.
- **Weight tying** — set `config.weightTying = true` to share embedding and output projection weights. The backward pass accumulates the projection gradient into the embedding gradient, keeping them in sync.
- **Gradient clipping** — `Optimizer:clipGradNorm(params, maxNorm)` scales all gradients by `maxNorm / globalNorm` when the global L2 norm exceeds `maxNorm`. Wired into the training loop via `TrainingOptions.gradClipNorm`.
- **LR scheduling** — `Optimizer:getLR(epoch, total, baseLR, warmupEpochs)` returns a linearly warmed-up then cosine-decayed learning rate. Enabled with `TrainingOptions.lrSchedule = true` and `TrainingOptions.warmupEpochs`.
- **KV-cache inference** — `MultiHeadAttention:forwardCached` and `clearKVCache` propagated up through `TransformerBlock` and `TransformerModel`. At inference time the cache grows one row per step rather than rebuilding the full attention matrix.
- **Top-k sampling** — `TransformerModel:predictNextTokenTopK(tokens, k, temperature)` and the high-level `LLMInstance:generateTopK(inputStr, numTokens, k, temperature?)`.
- **Nucleus (top-p) sampling** — `TransformerModel:predictNextTokenNucleus(tokens, p, temperature)` and `LLMInstance:generateNucleus(inputStr, numTokens, p, temperature?)`.
- **KV-cache generation** — `LLMInstance:generateWithKVCache(inputStr, numTokens, p?, temperature?)` seeds the cache with the prompt once, then appends one token at a time.
- **AdaptiveVocab** (`lib/AdaptiveVocab.luau`) — wraps any tokenizer and builds a dense vocab from the top-N most frequent tokens in the corpus. `encode` and `decode` map between text and dense IDs; OOV tokens map to a reserved UNK ID.
- **CorpusFetcher** (`lib/CorpusFetcher.luau`) — fetches text from HTTP URLs, splits into cleaned sentences, deduplicates, and optionally caches the result in DataStore to avoid re-downloading on restarts.
- **Multi-key DataStore save/load** — `saveModel` serializes weights to JSON and chunks it into ≤ 3.5 MB pieces stored under `storeKey_meta`, `storeKey_chunk0`, etc. `loadModel` reads the metadata key to discover chunk count, then reassembles. Falls back to the v0.1 single-key format when loading older saves.
- **`GenerativeBot.server.luau`** — new example in `examples/` demonstrating weight tying, grad clipping, LR scheduling, KV-cache generation, top-k, nucleus, corpus fetching, and DataStore save/load in one script.
- **Lune unit tests** — `tests/unit/` contains six test suites (LinearAlgebra, Optimizer, FeedForward, MultiHeadAttention, TransformerModel, AdaptiveVocab) runnable outside Roblox via `lune run tests/run_lune.luau`.
- **darklua integration** — `.darklua.json` transforms `require(script.Parent.X)` → relative file paths using the rojo sourcemap, making the source loadable by Lune without modifying production code.
- **CD workflow** (`.github/workflows/cd.yml`) — triggers on `v*` tags; runs the full validation pipeline, extracts the matching CHANGELOG section, creates a GitHub Release with the built `.rbxl` artifact, and publishes to Wally. Requires a `WALLY_AUTH_TOKEN` repository secret.

### Changed

- `FeedForward` — W1 and W2 now use Xavier normal init instead of `randomMatrix(0.01)`.
- `MultiHeadAttention` — Wq, Wk, Wv, Wo now use Xavier normal init.
- `TrainingOptions` — new fields: `batchSize`, `gradClipNorm`, `lrSchedule`, `warmupEpochs`, `onEpochComplete`.
- `TransformerConfig` — new optional field: `weightTying`.
- `SmokeTests.server.luau` — extended with assertions for all v0.2 additions.
- `ci.yml` — two new steps appended: `darklua process src dist` and `lune run tests/run_lune.luau`.
- `aftman.toml` — added `lune@0.8.9` and `darklua@0.13.1`.
- README — updated to document all v0.2 API, new Quick Start example, updated module table, performance table, and examples table.

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

[0.2.0]: https://github.com/rustyspottedcatt/RoLLM/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rustyspottedcatt/RoLLM/releases/tag/v0.1.0
