<img width="1500" height="500" alt="image" src="https://github.com/user-attachments/assets/d34446b8-0a26-4015-83cf-4ee2af32b075" />

# RoLLM

[![CI](https://github.com/rustyspottedcatt/RoLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/rustyspottedcatt/RoLLM/actions/workflows/ci.yml)
[![Wally](https://img.shields.io/badge/wally-rustyspottedcatt%2Frollm-orange)](https://wally.run/package/rustyspottedcatt/rollm)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Maintainer](https://img.shields.io/badge/maintainer-rustyspotted-blue)](https://github.com/rustyspottedcatt)

RoLLM is a lightweight transformer-based language model framework for the Roblox/Luau ecosystem. It ships a complete stack: character and BPE tokenizers, sinusoidal positional embeddings, multi-head causal attention, layer norm, feedforward blocks, Adam optimizer, cross-entropy loss, and temperature-sampled generation — all pure Luau, zero external dependencies beyond Wally packages.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [RoLLM.new](#rollmnew)
  - [trainModel](#trainmodel)
  - [generate / generateTemperature](#generate--generatetemperature)
  - [predict / predictTemperature](#predict--predicttemperature)
  - [getParameterCount](#getparametercount)
- [Configuration](#configuration)
- [TrainingOptions](#trainingoptions)
- [Modules](#modules)
- [Tokenizer Modes](#tokenizer-modes)
- [Performance Notes](#performance-notes)
- [Examples](#examples)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)

---

## Features

- **Char/BPE Tokenizers** — two strategies; BPE supports external vocab via HTTP.
- **Sinusoidal Positional Embeddings** — pre-computed at construction, zero runtime cost.
- **Multi-Head Causal Attention** — scaled dot-product with causal mask caching.
- **Transformer Blocks** — residual connections, pre-norm LayerNorm, ReLU FFN.
- **Adam Optimizer** — first- and second-moment estimates with bias correction.
- **Cross-Entropy Loss** — numerically stable softmax, full backward pass.
- **Temperature Generation** — greedy (`temperature ≈ 0`) and sampled decoding.
- **Async-friendly Training** — `yieldEverySamples` option prevents script timeout.
- **Parameter Count** — `getParameterCount()` for model introspection.
- **Fully Typed** — exported Luau types for all public surfaces.

---

## Architecture

```
Input Text
    │
    ▼
Tokenizer (char | bpe)
    │  builds vocab, textToTokens / tokensToText
    ▼
Embedding  (vocabSize × dModel)  +  Sinusoidal Positions
    │
    ▼  ×numLayers
TransformerBlock
  ├─ MultiHeadAttention  (causal mask, scaled dot-product)
  ├─ Residual + LayerNorm
  ├─ FeedForward  (Linear → ReLU → Linear)
  └─ Residual + LayerNorm
    │
    ▼
Final Projection  (dModel × vocabSize)
    │
    ▼
Logits  →  softmax  →  next-token prediction / sampling
```

---

## Installation

### Wally (recommended)

Add to your `wally.toml`:

```toml
[dependencies]
RoLLM = "rustyspottedcatt/rollm@0.1.0"
```

Then run:

```sh
wally install
```

### Manual (Rojo)

1. Clone the repo and place `src/` under `ReplicatedStorage/RoLLM` in your Rojo project tree, or use the provided `default.project.json`.
2. Run `wally install` to fetch the `promise` and `signal` dependencies into `Packages/`.

```
RoLLM/
├── components/
│   ├── BPETokenizer.luau
│   ├── CharTokenizer.luau
│   ├── Embedding.luau
│   ├── FeedForward.luau
│   ├── LayerNorm.luau
│   ├── LinearAlgebra.luau
│   ├── MultiHeadAttention.luau
│   ├── Tokenizer.luau
│   ├── TransformerBlock.luau
│   └── TransformerModel.luau
├── lib/
│   ├── CrossEntropyLoss.luau
│   ├── Optimizer.luau
│   └── types.luau
└── init.luau
```

---

## Quick Start

```lua
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RoLLM = require(ReplicatedStorage.RoLLM)

local corpus = {
    "hello roblox",
    "hello rollm",
    "roblox lua",
}

local model = RoLLM.new(corpus, {
    dModel        = 8,
    numHeads      = 2,
    dFF           = 16,
    numLayers     = 1,
    maxSeqLen     = 16,
    tokenizerMode = "char",
})

local history = model:trainModel(corpus, 3, 0.01)
print("Loss:", history[#history].averageLoss)
print("Params:", model:getParameterCount())
print(model:generate("hel", 10))
print(model:generateTemperature("rob", 10, 0.8))
```

---

## API Reference

### `RoLLM.new`

```lua
RoLLM.new(
    textData  : string | {string},
    config    : TransformerConfig,
    chunkSize : number?            -- default 500 000
) -> LLMInstance
```

Builds the tokenizer vocabulary from `textData`, constructs the model, and returns the LLM instance. Raises on invalid config.

---

### `trainModel`

```lua
model:trainModel(
    trainingData : {string},
    epochs       : number,
    learningRate : number,
    options      : TrainingOptions?
) -> {TrainingEpoch}
```

Trains the model and returns a history table. Each entry is `{ epoch, loss, averageLoss, samples }`.

---

### `generate` / `generateTemperature`

```lua
model:generate(inputStr: string, numTokens: number) -> string
model:generateTemperature(inputStr: string, numTokens: number, temperature: number) -> string
```

Extends `inputStr` by `numTokens` tokens. `generate` uses greedy decoding; `generateTemperature` samples from the softmax distribution scaled by `temperature` (lower = sharper, `0` = greedy).

---

### `predict` / `predictTemperature`

```lua
model:predict(inputStr: string) -> string
model:predictTemperature(inputStr: string, temperature: number) -> string
```

Returns the single most-likely next token as a string.

---

### `getParameterCount`

```lua
model:getParameterCount() -> number
```

Returns the total number of trainable scalar parameters (embedding + all block weights + final projection).

---

## Configuration

`TransformerConfig` fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `dModel` | `number` | ✓ | Embedding / hidden dimension. Must be divisible by `numHeads`. |
| `numHeads` | `number` | ✓ | Number of attention heads. |
| `dFF` | `number` | ✓ | Feedforward inner dimension. |
| `numLayers` | `number` | ✓ | Number of transformer blocks. |
| `maxSeqLen` | `number` | ✓ | Maximum token sequence length. |
| `tokenizerMode` | `"char"\|"bpe"` | | Default `"char"`. |
| `externalVocabURL` | `string` | | HTTP URL for BPE vocab JSON. Required when `tokenizerMode = "bpe"`. |
| `vocabSize` | `number` | | Set automatically; do not pass manually. |

---

## TrainingOptions

```lua
type TrainingOptions = {
    yieldEverySamples         : number?,  -- call task.wait() every N samples
    yieldEveryParameterUpdates: number?,  -- reserved for future use
}
```

Use `yieldEverySamples = 1` in long-running server scripts to avoid the Roblox script timeout.

---

## Modules

| Module | Path | Role |
|---|---|---|
| `RoLLM` | `src/init.luau` | Public API, training loop |
| `TransformerModel` | `components/TransformerModel.luau` | Forward / backward, position/mask caches |
| `TransformerBlock` | `components/TransformerBlock.luau` | One transformer block (MHA + FFN + norms) |
| `MultiHeadAttention` | `components/MultiHeadAttention.luau` | Scaled dot-product, causal masking |
| `FeedForward` | `components/FeedForward.luau` | Two-layer ReLU FFN |
| `LayerNorm` | `components/LayerNorm.luau` | Per-row layer normalisation |
| `Embedding` | `components/Embedding.luau` | Token + sinusoidal position embeddings |
| `LinearAlgebra` | `components/LinearAlgebra.luau` | Matrix ops (mm, add, softmax, relu, …) |
| `Tokenizer` | `components/Tokenizer.luau` | Mode factory |
| `CharTokenizer` | `components/CharTokenizer.luau` | Character-level tokenizer |
| `BPETokenizer` | `components/BPETokenizer.luau` | BPE tokenizer with external vocab |
| `CrossEntropyLoss` | `lib/CrossEntropyLoss.luau` | Stable softmax CE loss + backward |
| `Optimizer` | `lib/Optimizer.luau` | Adam optimizer |
| `types` | `lib/types.luau` | Shared Luau type exports |

---

## Tokenizer Modes

- **`"char"`** (default) — builds a vocabulary of individual characters from the training corpus. Fast, portable, no external data needed.
- **`"bpe"`** — Byte-Pair Encoding. Requires an external JSON vocab loaded with `config.externalVocabURL`. Provides better sub-word generalisation.

---

## Performance Notes

The runtime applies several optimizations to stay Roblox-friendly:

- **Causal mask cache** — the `seqLen × seqLen` mask is built once per distinct sequence length and reused.
- **Position index cache** — position arrays are cached by length, never re-allocated.
- **Training tokenization** — all training samples are tokenized once before the epoch loop.
- **Transposed B in `mm`** — `LinearAlgebra:mm` pre-transposes the right-hand matrix so the inner loop is row-major, improving sequential table access in Luau.
- **Attention backward** — the softmax backward reuses cached `attn` weights instead of recomputing `math.exp`.

Keep `dModel`, `dFF`, `numLayers`, and `maxSeqLen` small and scale gradually. A model with `dModel=8, dFF=16, numLayers=1` runs comfortably in Studio.

---

## Examples

| Script | Location | Description |
|---|---|---|
| Tiny training | `examples/TinyTraining.server.luau` | Minimal train-and-generate loop |
| In-game chatbot | `examples/InGameChatBot.server.luau` | Chat-triggered bot (`!rollm <prompt>`) |
| Smoke tests | `tests/SmokeTests.server.luau` | Tokenization, loss, epoch, prediction |

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. For bugs use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml) and for features use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml).

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Security

To report a vulnerability privately, see [SECURITY.md](SECURITY.md).

---

## License

Distributed under the [MIT License](LICENSE).  
© 2025-2026 [rustyspotted](https://github.com/rustyspottedcatt)
