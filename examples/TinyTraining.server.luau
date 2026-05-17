--[[
    TinyTraining.server.luau
    
    The simplest possible RoLLM usage: define a small corpus, train a minimal
    model for a few epochs, then run greedy and temperature-sampled generation.
    
    Good starting point before moving to larger examples.
]]

local Config = require(script.Parent._Config)
if not Config[script.Name] then return end

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RoLLM = require(ReplicatedStorage.RoLLM)

local corpus = {
	"hello world",
	"hello roblox",
	"roblox is fun",
	"hello hello hello",
	"fun fun fun world",
}

local llm = RoLLM.new(corpus, {
	dModel = 16,
	numHeads = 2,
	dFF = 32,
	numLayers = 1,
	maxSeqLen = 16,
	tokenizerMode = "char",
})

print(string.format("Model created. Parameters: %d", llm:getParameterCount()))
print(string.format("Vocab size: %d", llm._tokenizer:getVocabSize()))

local history = llm:trainModel(corpus, 20, 0.01, {
	yieldEverySamples = 1,
	onEpochComplete = function(epoch, total, avgLoss, elapsed)
		if epoch % 5 == 0 then
			print(string.format("Epoch %d/%d | Loss: %.4f | Elapsed: %.1fs", epoch, total, avgLoss, elapsed))
		end
	end,
})

print(string.format("Training complete. Final loss: %.4f", history[#history].averageLoss))

print("--- Greedy generation ---")
print(llm:generate("hello", 20))

print("--- Temperature generation (0.8) ---")
print(llm:generateTemperature("hello", 20, 0.8))

print("--- Temperature generation (1.2) ---")
print(llm:generateTemperature("roblox", 20, 1.2))
