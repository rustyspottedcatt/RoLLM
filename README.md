<img width="1500" height="500" alt="image" src="https://github.com/user-attachments/assets/d34446b8-0a26-4015-83cf-4ee2af32b075" />

# RoLLM

[![CI](https://github.com/rustyspottedcatt/RoLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/rustyspottedcatt/RoLLM/actions/workflows/ci.yml)
[![CD](https://github.com/rustyspottedcatt/RoLLM/actions/workflows/cd.yml/badge.svg)](https://github.com/rustyspottedcatt/RoLLM/actions/workflows/cd.yml)
[![Wally](https://img.shields.io/badge/wally-rustyspottedcatt%2Frollm-orange)](https://wally.run/package/rustyspottedcatt/rollm)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Maintainer](https://img.shields.io/badge/maintainer-rustyspotted-blue)](https://github.com/rustyspottedcatt)

A transformer language model framework for Roblox/Luau. The full stack — tokenizers, positional embeddings, multi-head attention, layer norm, feedforward blocks, Adam, cross-entropy — in pure Luau. No native code, no external runtime, no magic.

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
  - [generateTopK / generateNucleus / generateWithKVCache](#generatetopk--generatenucleus--generatewithkvcache)
  - [predict / predictTemperature](#predict--predicttemperature)
  - [saveModel / loadModel](#savemodel--loadmodel)
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

**Core (v0.1)**
- **Char/BPE Tokenizers** — char builds vocab from the corpus directly; BPE loads an external JSON vocab over HTTP.
- **Sinusoidal Positional Embeddings** — computed once at construction, not every forward pass.
- **Multi-Head Causal Attention** — scaled dot-product with a cached causal mask.
- **Transformer Blocks** — pre-norm LayerNorm, residual connections.
- **Adam Optimizer** — bias-corrected first and second moment estimates.
- **Cross-Entropy Loss** — numerically stable softmax, full backward pass.
- **Temperature Sampling** — greedy (`temperature = 0`) or distribution-sampled decoding.
- **Async Training** — `yieldEverySamples` stops Roblox from timing out long training runs.

**Generative (v0.2)**
- **GELU Activation** — replaces ReLU in the feedforward block; exact `tanh` formulation with a matching backward.
- **Xavier Initialization** — all weight matrices (`Wq`, `Wk`, `Wv`, `Wo`, `W1`, `W2`) initialized with Xavier normal.
- **Weight Tying** — set `weightTying = true` to share embedding and output projection weights, cutting parameter count.
- **Gradient Clipping** — `gradClipNorm` in `TrainingOptions` clips the global L2 gradient norm before each Adam step.
- **LR Scheduling** — linear warmup then cosine decay via `lrSchedule = true` and `warmupEpochs`.
- **Top-k Sampling** — `generateTopK` samples from the top `k` logits after temperature scaling.
- **Nucleus Sampling** — `generateNucleus` samples from the smallest set of tokens whose cumulative probability ≥ `p`.
- **KV-Cache Generation** — `generateWithKVCache` seeds the cache with the prompt once, then appends one token at a time.
- **AdaptiveVocab** — maps a large BPE vocabulary down to a smaller dense vocab built from actual corpus token frequencies.
- **CorpusFetcher** — fetches and caches text corpora from HTTP URLs into DataStore so you don't re-download on every restart.
- **Multi-Key DataStore** — `saveModel` / `loadModel` chunk the serialized weights into ≤ 3.5 MB pieces to work around DataStore limits.

---

## Architecture

```
Input Text
    │
    ▼
Tokenizer (char | bpe)  ──optional──▶  AdaptiveVocab (dense remapping)
    │  textToTokens / tokensToText
    ▼
Embedding  (vocabSize × dModel, Xavier init)  +  Sinusoidal Positions
    │                                              ↑ shared when weightTying = true
    ▼  ×numLayers                                  │
TransformerBlock                                   │
  ├─ MultiHeadAttention  (causal mask, KV-cache for inference)
  ├─ Residual + LayerNorm
  ├─ FeedForward  (Linear → GELU → Linear, Xavier init)
  └─ Residual + LayerNorm
    │
    ▼
Final Projection  (dModel × vocabSize)
    │
    ▼
Logits  →  greedy | temperature | top-k | nucleus sampling
```

---

## Installation

### Wally (recommended)

```toml
[dependencies]
RoLLM = "rustyspottedcatt/rollm@0.2.0"
```

```sh
wally install
```

### Manual (Rojo)

Clone the repo and point `src/` at `ReplicatedStorage/RoLLM` in your project tree, or use the provided `default.project.json`. Run `wally install` to pull in the `promise` and `signal` dependencies.

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
│   ├── AdaptiveVocab.luau
│   ├── CorpusFetcher.luau
│   ├── CrossEntropyLoss.luau
│   ├── Optimizer.luau
│   └── types.luau
└── init.luau
```

---

## Quick Start

### Basic (v0.1 style, still works)

```lua
local RoLLM = require(game:GetService("ReplicatedStorage").RoLLM)

local corpus = { "hello roblox", "hello rollm", "roblox lua" }

local model = RoLLM.new(corpus, {
    dModel        = 8,
    numHeads      = 2,
    dFF           = 16,
    numLayers     = 1,
    maxSeqLen     = 16,
    tokenizerMode = "char",
})

model:trainModel(corpus, 3, 0.01)
print(model:generate("hel", 10))
```

### Generative (v0.2)

```lua
local RoLLM = require(game:GetService("ReplicatedStorage").RoLLM)

local corpus = { "press space to jump", "collect coins for points", "defeat the boss" }

local model = RoLLM.new(corpus, {
    dModel        = 128,
    numHeads      = 4,
    dFF           = 256,
    numLayers     = 4,
    maxSeqLen     = 64,
    tokenizerMode = "char",
    weightTying   = true,
})

model:trainModel(corpus, 200, 3e-4, {
    gradClipNorm = 1.0,
    lrSchedule   = true,
    warmupEpochs = 10,
})

-- Three ways to generate:
print(model:generateTopK("press", 20, 40, 0.9))
print(model:generateNucleus("collect", 20, 0.95, 0.85))
print(model:generateWithKVCache("defeat", 20, 0.9, 0.7))

model:saveModel("MyBotWeights")
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

Builds the tokenizer vocab from `textData`, constructs the model, and returns the instance. Raises if `dModel` is not divisible by `numHeads` or if required config fields are missing.

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

Returns a history table where each entry is `{ epoch, loss, averageLoss, samples }`.

---

### `generate` / `generateTemperature`

```lua
model:generate(inputStr: string, numTokens: number) -> string
model:generateTemperature(inputStr: string, numTokens: number, temperature: number) -> string
```

Greedy decoding and temperature-sampled decoding. `temperature = 0` is greedy; higher values flatten the distribution.

---

### `generateTopK` / `generateNucleus` / `generateWithKVCache`

```lua
model:generateTopK(inputStr: string, numTokens: number, k: number, temperature: number?) -> string
model:generateNucleus(inputStr: string, numTokens: number, p: number, temperature: number?) -> string
model:generateWithKVCache(inputStr: string, numTokens: number, p: number?, temperature: number?) -> string
```

- **`generateTopK`** — samples from the top `k` logits after temperature scaling.
- **`generateNucleus`** — samples from the smallest set of tokens whose cumulative probability ≥ `p`.
- **`generateWithKVCache`** — nucleus sampling with a KV-cache. Seeds the cache with the full prompt once, then appends one token per step. Faster for long outputs because each new token only processes one row instead of the full sequence.

---

### `predict` / `predictTemperature`

```lua
model:predict(inputStr: string) -> string
model:predictTemperature(inputStr: string, temperature: number) -> string
```

Returns the single next-token prediction as a string.

---

### `saveModel` / `loadModel`

```lua
model:saveModel(storeKey: string) -> boolean
model:loadModel(storeKey: string) -> boolean
```

Serializes model weights to JSON, splits into ≤ 3.5 MB chunks, and stores them under `storeKey_meta` + `storeKey_chunk0`, `storeKey_chunk1`, etc. `loadModel` reads the metadata key first, then reassembles. Returns `false` on any DataStore error. Requires `DataStoreService` access.

---

### `getParameterCount`

```lua
model:getParameterCount() -> number
```

Total trainable scalar parameters across embedding, all blocks, and the final projection.

---

## Configuration

`TransformerConfig` fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `dModel` | `number` | ✓ | Hidden dimension. Must be divisible by `numHeads`. |
| `numHeads` | `number` | ✓ | Number of attention heads. |
| `dFF` | `number` | ✓ | Feedforward inner dimension. |
| `numLayers` | `number` | ✓ | Number of transformer blocks. |
| `maxSeqLen` | `number` | ✓ | Maximum token sequence length. Inputs longer than this are trimmed. |
| `tokenizerMode` | `"char"\|"bpe"` | | Default `"char"`. |
| `externalVocabURL` | `string` | | BPE vocab JSON URL. Required when `tokenizerMode = "bpe"`. |
| `weightTying` | `boolean` | | Share embedding and output projection weights. Default `false`. |
| `vocabSize` | `number` | | Set automatically; do not pass manually. |

---

## TrainingOptions

```lua
type TrainingOptions = {
    batchSize                 : number?,   -- gradient accumulation batch size
    yieldEverySamples         : number?,   -- call task.wait() every N samples
    yieldEveryParameterUpdates: number?,   -- reserved
    gradClipNorm              : number?,   -- max global L2 gradient norm (e.g. 1.0)
    lrSchedule                : boolean?,  -- enable linear warmup + cosine decay
    warmupEpochs              : number?,   -- epochs for the warmup phase
    onEpochComplete           : ((epoch: number, total: number, avgLoss: number, elapsed: number) -> ())?,
}
```

`yieldEverySamples = 1` is still the safest option to avoid script timeouts in long runs. Combine `gradClipNorm = 1.0` with `lrSchedule = true` for more stable training on larger models.

---

## Modules

| Module | Path | Role |
|---|---|---|
| `RoLLM` | `src/init.luau` | Public API, training loop |
| `TransformerModel` | `components/TransformerModel.luau` | Forward / backward, sampling, KV-cache |
| `TransformerBlock` | `components/TransformerBlock.luau` | MHA + FFN + residuals + norms |
| `MultiHeadAttention` | `components/MultiHeadAttention.luau` | Scaled dot-product, causal mask, KV-cache |
| `FeedForward` | `components/FeedForward.luau` | Two-layer GELU FFN, Xavier init |
| `LayerNorm` | `components/LayerNorm.luau` | Per-row layer normalization |
| `Embedding` | `components/Embedding.luau` | Token embeddings + sinusoidal positions |
| `LinearAlgebra` | `components/LinearAlgebra.luau` | Matrix ops (mm, softmax, xavier, GELU, …) |
| `Tokenizer` | `components/Tokenizer.luau` | Mode factory (char / bpe) |
| `CharTokenizer` | `components/CharTokenizer.luau` | Character-level tokenizer |
| `BPETokenizer` | `components/BPETokenizer.luau` | BPE tokenizer with external vocab |
| `AdaptiveVocab` | `lib/AdaptiveVocab.luau` | Remaps a large BPE vocab to a smaller dense vocab |
| `CorpusFetcher` | `lib/CorpusFetcher.luau` | HTTP corpus fetch with DataStore sentence cache |
| `CrossEntropyLoss` | `lib/CrossEntropyLoss.luau` | Stable softmax CE loss + backward |
| `Optimizer` | `lib/Optimizer.luau` | Adam + grad clipping + LR scheduling |
| `types` | `lib/types.luau` | Shared Luau type exports |

---

## Tokenizer Modes

- **`"char"`** (default) — one token per character, vocab built entirely from the training text. Nothing to download, works offline.
- **`"bpe"`** — Byte-Pair Encoding. Needs an external JSON vocab at `config.externalVocabURL`. Better sub-word coverage for natural language, but requires HTTP access.

You can layer **AdaptiveVocab** on top of either tokenizer to shrink a large vocab down to the tokens that actually appear in your corpus. Useful if you're loading a GPT-2–style vocab but only training on game-dialogue sentences.

---

## Performance Notes

- **Causal mask cache** — the `seqLen × seqLen` mask is built once per distinct sequence length and reused.
- **Position index cache** — position arrays are cached by length, never re-allocated.
- **Training tokenization** — all samples tokenize before the epoch loop, not inside it.
- **Row-major `mm`** — `LinearAlgebra:mm` pre-transposes the right operand so the inner loop hits sequential memory.
- **Attention backward** — reuses the cached softmax weights rather than recomputing `math.exp`.
- **KV-cache** — at inference time `forwardCached` grows the K/V tables one row at a time instead of rebuilding the full attention every step.

Rough Studio numbers to calibrate expectations:

| Config | Params | Train (1 epoch / 20 samples) |
|---|---|---|
| `dModel=8, dFF=16, L=1` | ~500 | < 0.1s |
| `dModel=64, dFF=128, L=2` | ~85k | ~1–2s |
| `dModel=128, dFF=256, L=4` | ~430k | ~8–15s |

Set `yieldEverySamples` to something small (1–5) for the larger configs.

---

## Examples

| Script | Location | Description |
|---|---|---|
| Tiny training | `examples/TinyTraining.server.luau` | Minimal train-and-generate loop |
| In-game chatbot | `examples/InGameChatBot.server.luau` | Chat bot responding to `!rollm <prompt>` |
| Generative bot | `examples/GenerativeBot.server.luau` | Full v0.2 demo: weight tying, grad clipping, LR schedule, KV-cache, top-k, nucleus, DataStore save/load |
| Smoke tests | `tests/SmokeTests.server.luau` | Roblox-runtime assertions covering v0.1 and v0.2 |
| Unit tests | `tests/unit/` | Lune-runnable pure-math tests (run via `lune run tests/run_lune.luau` after `darklua process src dist`) |

---

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml) for bugs and the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml) for new ideas.

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Security

Report vulnerabilities privately via [SECURITY.md](SECURITY.md).

---

## License

[MIT License](LICENSE) — © 2025-2026 [rustyspotted](https://github.com/rustyspottedcatt)
