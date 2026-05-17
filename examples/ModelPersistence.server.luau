--[[
    ModelPersistence.server.luau
    
    Demonstrates saving and loading trained model weights with DataStoreService.
    
    First run:  trains the model, saves weights under STORE_KEY, prints inference.
    Next runs:  loads weights instantly, skips training entirely.
    
    Change STORE_KEY whenever you modify the corpus or model config — the stored
    weights are tied to a specific architecture and vocab.
    
    Requirements:
      Studio:     Game Settings > Security > Enable Studio Access to API Services
      Live game:  works automatically
]]

local Config = require(script.Parent._Config)
if not Config[script.Name] then return end

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RoLLM = require(ReplicatedStorage.RoLLM)

local STORE_KEY = "RoLLM_Example_v1"

local corpus = {
	"the quick brown fox jumps over the lazy dog",
	"to be or not to be that is the question",
	"all that glitters is not gold",
	"the only thing we have to fear is fear itself",
	"ask not what your country can do for you",
	"in the beginning was the word",
	"it was the best of times it was the worst of times",
}

local llm = RoLLM.new(corpus, {
	dModel = 32,
	numHeads = 4,
	dFF = 64,
	numLayers = 2,
	maxSeqLen = 32,
	tokenizerMode = "char",
})

print(string.format("[Persistence] Model: %d params, vocab: %d chars",
	llm:getParameterCount(), llm._tokenizer:getVocabSize()))

print(string.format("[Persistence] Checking DataStore for key '%s'...", STORE_KEY))

local loaded = llm:loadModel(STORE_KEY)

if loaded then
	print("[Persistence] Weights loaded. Skipping training.")
else
	print("[Persistence] No saved weights found. Training from scratch...")

	local history = llm:trainModel(corpus, 30, 0.01, {
		batchSize = 4,
		yieldEverySamples = 1,
		onEpochComplete = function(epoch, total, avgLoss, elapsed)
			local eta = if epoch < total then (elapsed / epoch) * (total - epoch) else 0
			local pct = math.floor((epoch / total) * 20)
			local bar = string.rep("=", pct) .. string.rep("-", 20 - pct)
			print(string.format("[Persistence] [%s] %d/%d | Loss: %.4f | ETA: %.0fs",
				bar, epoch, total, avgLoss, eta))
		end,
	})

	print(string.format("[Persistence] Training complete. Loss: %.4f", history[#history].averageLoss))

	local saved = llm:saveModel(STORE_KEY)
	if saved then
		print(string.format("[Persistence] Weights saved to DataStore key '%s'.", STORE_KEY))
		print("[Persistence] Restart the server — it will load instantly next time.")
	else
		warn("[Persistence] Save failed. Check that API Services are enabled in Studio.")
	end
end

print("--- Inference ---")
local seeds = { "the ", "to b", "all ", "ask " }
for _, seed in ipairs(seeds) do
	local result = llm:generateTemperature(seed, 30, 0.6)
	print(string.format("  seed %q -> %q", seed, result))
end
