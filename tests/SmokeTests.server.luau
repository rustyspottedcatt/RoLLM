local ReplicatedStorage = game:GetService("ReplicatedStorage")

local RoLLM = require(ReplicatedStorage.RoLLM)

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

print("RoLLM smoke tests passed")
