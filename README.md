<img src="https://l7mozmkiwy.ufs.sh/f/HKemhjN71TyOBDSBjRYWE0OaYPF9Vq4jUDItmN6JuXrkiTAe" alt="NEBYTE">

# RoLLM (Early Alpha Release v0.1.0)

[![Maintainer](https://img.shields.io/badge/maintainer-rustyspotted-blue)](https://github.com/rustyspottedcatt)
[![Made with Lua](https://img.shields.io/badge/Made%20with-Lua-000080)](https://www.lua.org/)
[![License](https://img.shields.io/badge/License-MIT-blue)](https://choosealicense.com/licenses/mit/)

RoLLM is a lightweight and educational transformer-based language model framework built entirely for the ROBLOX ecosystem. It includes a full list from character and BPE-based tokenizers to multi-head attention, training loops, and inference with temperature control.

> 🚧 **This project is in early alpha** and not intended for production usage or real deployment on ROBLOX experiences.

---

## Table of Contents

* [Features](#features)
* [Architecture Overview](#architecture-overview)
* [Installation](#installation)
* [Usage](#usage)

  * [Quick Start](#quick-start)
  * [Training](#training)
  * [Generation](#generation)
* [Modules](#modules)
* [Tokenizer Modes](#tokenizer-modes)
* [License](#license)

---

## Features

* **Char/BPE Tokenizers**: Two tokenizer strategies with external vocab support for BPE.
* **Custom Embedding Layer**: Position-aware token embedding matrices.
* **Multi-Head Attention**: Scaled dot-product attention with causal masking.
* **Transformer Blocks**: Includes layer norm, residual connections, and feedforward networks.
* **Training Loop**: Cross-entropy loss and basic SGD optimizer.
* **Generation**: Sampling and greedy decoding with temperature control.
* **Modular Design**: Swap out components easily (e.g., tokenizer, attention).

---

## Architecture Overview

```
Input Text → Tokenizer → Embedding → N × Transformer Blocks
             ↓                               ↓
        Vocabulary         ←         Final Projection (d_model × vocab_size)
```

---

## Installation

1. **Clone the Repo**
   Place the contents into `ReplicatedStorage` in your Roblox project.

```
RoLLM/
├── components/
│   ├── Tokenizer.luau
│   ├── CharTokenizer.luau
│   ├── BPETokenizer.luau
│   ├── Embedding.luau
│   ├── MultiHeadAttention.luau
│   ├── TransformerBlock.luau
│   ├── TransformerModel.luau
├── lib/
│   ├── LinearAlgebra.luau
│   ├── CrossEntropyLoss.luau
│   ├── Optimizer.luau
│   ├── types.luau
├── RoLLM.luau
```

2. **Require from any script:**

```lua
local RoLLM = require(ReplicatedStorage.RoLLM)
```

---

## Usage

### Quick Start

```lua
local RoLLM = require(script.Parent.RoLLM)

local model = RoLLM.new("hello world", {
    dModel = 64,
    numHeads = 4,
    dFF = 128,
    numLayers = 2,
    maxSeqLen = 64,
    tokenizerMode = "char"
})

print(model:generate("hel", 10))
```

### Training

```lua
model:trainModel({
    "hello world",
    "how are you",
    "i am a bot"
}, 5, 0.01)
```

### Generation

```lua
local output = model:generateTemperature("hello", 20, 0.8)
print(output)
```

---

## Modules

* `RoLLM.lua`: Entry point, builds the tokenizer and transformer.
* `Tokenizer.lua`: Factory for `char` or `bpe` modes.
* `CharTokenizer.lua`: Character-level tokenizer.
* `BPETokenizer.lua`: Byte-Pair Encoding tokenizer with external vocab support.
* `TransformerModel.lua`: Core forward/backward implementation.
* `Embedding.lua`: Position-encoded embeddings.
* `MultiHeadAttention.lua`: Attention module.
* `TransformerBlock.lua`: One transformer block.
* `LinearAlgebra.lua`: Basic matrix math.
* `CrossEntropyLoss.lua`: Computes loss and gradients.
* `Optimizer.lua`: Naive SGD optimizer.
* `types.lua`: Shared matrix and config types.

---

## Tokenizer Modes

* `"char"`: Default. Simple and fast but limited generalization.
* `"bpe"`: External vocab must be loaded with `loadExternalVocab(url)`.

---

## License

Distributed under the [MIT License](https://choosealicense.com/licenses/mit/).
© 2024 [rustyspotted](https://github.com/rustyspottedcatt) — All rights reserved.