--[[
    TemperatureShowcase.server.luau
    
    Trains a model then runs the same seed through multiple temperature values,
    showing how temperature controls the randomness of generation:
    
      temperature → 0   : greedy (always picks highest probability token)
      temperature = 0.5 : fairly focused, minor variation
      temperature = 1.0 : sample directly from the model distribution
      temperature > 1.0 : flatter distribution, more random/creative output
    
    Run this multiple times with the same trained model to see how each
    temperature behaves differently on repeated calls.
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RoLLM = require(ReplicatedStorage.RoLLM)

local GENERATE_LENGTH = 40
local RUNS_PER_TEMP = 3

local corpus = {
	"the cat sat on the mat",
	"the cat sat on the hat",
	"the dog sat on the log",
	"the cat and the dog sat",
	"sat the cat on the mat today",
	"the mat was flat and that was that",
	"a cat a hat a mat a bat",
	"the cat is on the mat now",
}

local llm = RoLLM.new(corpus, {
	dModel = 24,
	numHeads = 4,
	dFF = 48,
	numLayers = 2,
	maxSeqLen = 24,
	tokenizerMode = "char",
})

print(string.format("Training... (%d params)", llm:getParameterCount()))

llm:trainModel(corpus, 40, 0.01, {
	batchSize = 4,
	yieldEverySamples = 1,
	onEpochComplete = function(epoch, total, avgLoss, _elapsed)
		if epoch == total then
			print(string.format("Training done. Final loss: %.4f", avgLoss))
		end
	end,
})

local temperatures = { 0.0, 0.5, 0.8, 1.0, 1.5 }
local seed = "the cat"

print(string.format("\nSeed: %q  |  Generate length: %d  |  Runs per temp: %d\n",
	seed, GENERATE_LENGTH, RUNS_PER_TEMP))
print(string.rep("-", 60))

for _, temp in ipairs(temperatures) do
	if temp == 0.0 then
		print("temperature = 0.0  (greedy)")
		for _ = 1, RUNS_PER_TEMP do
			print(string.format("  %q", llm:generate(seed, GENERATE_LENGTH)))
		end
	else
		print(string.format("temperature = %.1f", temp))
		for _ = 1, RUNS_PER_TEMP do
			print(string.format("  %q", llm:generateTemperature(seed, GENERATE_LENGTH, temp)))
		end
	end
	print(string.rep("-", 60))
end
