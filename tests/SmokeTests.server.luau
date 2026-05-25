local ReplicatedStorage = game:GetService("ReplicatedStorage")

local RoLLM = require(ReplicatedStorage.RoLLM) :: any

local function assertEqual(actual, expected, message)
	assert(actual == expected, string.format("%s: expected %s, got %s", message, tostring(expected), tostring(actual)))
end

local function assertThrows(callback, message)
	local ok = pcall(callback)
	assert(not ok, message)
end

local tokenizer = RoLLM.CharTokenizer.new()
tokenizer:buildVocabFromText("abc")
assert(tokenizer:getVocabSize() >= 4, "char tokenizer should include vocabulary and unknown token")
assertEqual(tokenizer:tokensToText(tokenizer:textToTokens("abc")), "abc", "char tokenizer should round-trip known text")

local loss = RoLLM.CrossEntropyLoss.new()
local value = loss:compute({ { 3, 1 }, { 1, 3 } }, { 1, 2 })
assert(value > 0, "cross entropy loss should be positive")
local grad = loss:backward({ { 3, 1 }, { 1, 3 } }, { 1, 2 })
assert(#grad == 2 and #grad[1] == 2, "cross entropy gradient should match logits shape")

local optimizer = RoLLM.Optimizer.new(0.01)
local weights = { { 1, 2 }, { 3, 4 } }
local updates = optimizer:updateParameters({
	weights = {
		value = weights,
		grad = { { 0.1, 0.1 }, { 0.1, 0.1 } },
	},
})
assertEqual(updates, 4, "optimizer should report matrix update count")
assert(weights[1][1] < 1, "optimizer should update matrix parameters")

local vector = { 1, 2, 3 }
local vectorUpdates = optimizer:updateParameters({
	vector = {
		value = vector,
		grad = { 0.1, 0.1, 0.1 },
	},
})
assertEqual(vectorUpdates, 3, "optimizer should report vector update count")
assert(vector[1] < 1, "optimizer should update vector parameters")

local corpus = { "hello", "help" }
local llm = RoLLM.new(corpus, {
	dModel = 8,
	numHeads = 2,
	dFF = 16,
	numLayers = 1,
	maxSeqLen = 8,
	tokenizerMode = "char",
})

local history = llm:trainModel(corpus, 1, 0.01)
assert(#history == 1, "trainModel should return one history entry per epoch")
assert(history[1].samples == #corpus, "trainModel should report sample count")
assert(llm:getParameterCount() > 0, "LLM should expose a positive parameter count")
assert(type(llm:predict("h")) == "string", "predict should return a string")
assert(#llm:generate("h", 2) >= 1, "generate should return text")
assert(#llm:generate("hellohellohello", 2) >= 1, "generate should safely truncate long contexts")

local yieldedHistory = llm:trainModel(corpus, 1, 0.01, {
	yieldEverySamples = 1,
	yieldEveryParameterUpdates = 100,
})
assertEqual(#yieldedHistory, 1, "yield-friendly training should return history")

assertThrows(function()
	RoLLM.new("abc", {
		dModel = 7,
		numHeads = 2,
		dFF = 16,
		numLayers = 1,
		maxSeqLen = 8,
		tokenizerMode = "char",
	})
end, "invalid head configuration should fail")

assertThrows(function()
	llm:trainModel({ "h" }, 1, 0.01)
end, "too-short training samples should fail")

-- ── v2: Optimizer extensions ──────────────────────────────────────────────────
local opt2 = RoLLM.Optimizer.new(0.01)
local bigParams = {
	w = { value = { { 100.0, 100.0 } }, grad = { { 100.0, 100.0 } } },
}
opt2:clipGradNorm(bigParams, 1.0)
local norm = math.sqrt(bigParams.w.grad[1][1] ^ 2 + bigParams.w.grad[1][2] ^ 2)
assert(norm <= 1.0 + 1e-5, "clipGradNorm should reduce norm to ≤ maxNorm")

local lrEarly = opt2:getLR(1, 100, 0.01, 10)
local lrMid   = opt2:getLR(10, 100, 0.01, 10)
local lrLate  = opt2:getLR(100, 100, 0.01, 10)
assert(lrEarly < lrMid, "warmup LR should increase over time")
assert(lrLate  < lrMid, "cosine decay LR should decrease after peak")

-- ── v2: weight tying ─────────────────────────────────────────────────────────
local tiedLLM = RoLLM.new({ "hello world", "test abc" }, {
	dModel      = 8,
	numHeads    = 2,
	dFF         = 16,
	numLayers   = 1,
	maxSeqLen   = 8,
	tokenizerMode = "char",
	weightTying = true,
})
assert(tiedLLM._model.weightTied == true, "weightTied flag should be set")

-- ── v2: grad clipping + LR schedule wired through trainModel ─────────────────
local clipHistory = llm:trainModel(corpus, 2, 0.01, {
	gradClipNorm = 1.0,
	lrSchedule   = true,
	warmupEpochs = 1,
})
assert(#clipHistory == 2, "grad-clip+LR-schedule training should return history")
assert(clipHistory[1].averageLoss ~= nil, "history should contain averageLoss")

-- ── v2: top-k generation ─────────────────────────────────────────────────────
local topkResult = llm:generateTopK("hel", 3, 5, 1.0)
assert(type(topkResult) == "string", "generateTopK should return a string")
assert(#topkResult >= 1, "generateTopK should return non-empty text")

-- ── v2: nucleus generation ────────────────────────────────────────────────────
local nucleusResult = llm:generateNucleus("hel", 3, 0.9, 0.8)
assert(type(nucleusResult) == "string", "generateNucleus should return a string")
assert(#nucleusResult >= 1, "generateNucleus should return non-empty text")

-- ── v2: KV-cache generation ───────────────────────────────────────────────────
local kvResult = llm:generateWithKVCache("hel", 4, 0.9, 0.7)
assert(type(kvResult) == "string", "generateWithKVCache should return a string")
assert(#kvResult >= 1, "generateWithKVCache should return non-empty text")

-- Ensure KV-cache is cleared after generation (model still usable)
local postKV = llm:predict("h")
assert(type(postKV) == "string", "model should still work after KV-cache generation")

-- ── v2: predictNextTokenTopK / Nucleus on model directly ─────────────────────
local topkId = llm._model:predictNextTokenTopK(
	llm._tokenizer:textToTokens("hel"), 5, 1.0
)
assert(topkId >= 1 and topkId <= llm._model.vocabSize,
	"predictNextTokenTopK should return a valid token id")

local nucId = llm._model:predictNextTokenNucleus(
	llm._tokenizer:textToTokens("hel"), 0.9, 1.0
)
assert(nucId >= 1 and nucId <= llm._model.vocabSize,
	"predictNextTokenNucleus should return a valid token id")

-- ── v2: forwardCached + clearKVCache ─────────────────────────────────────────
local m = llm._model
m:clearKVCache()
local seedTokens = llm._tokenizer:textToTokens("hel")
local cachedOut  = m:forwardCached(seedTokens, 1)
assert(#cachedOut == #seedTokens, "forwardCached output rows should equal seqLen")
assert(#cachedOut[1] == m.vocabSize, "forwardCached output cols should equal vocabSize")
m:clearKVCache()

-- ── v2: AdaptiveVocab smoke ───────────────────────────────────────────────────
local av = RoLLM.AdaptiveVocab.new(llm._tokenizer, 32)
av:buildFromCorpus(corpus)
assert(av:getVocabSize() > 0, "AdaptiveVocab size should be > 0")
assert(av:getVocabSize() <= 32, "AdaptiveVocab size should be capped at maxVocab")
local encoded = av:encode("hel")
assert(type(encoded) == "table" and #encoded > 0, "encode should return token ids")
local decoded = av:decode(encoded)
assert(type(decoded) == "string", "decode should return a string")

print("RoLLM smoke tests passed")
