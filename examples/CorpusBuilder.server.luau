--[[
    CorpusBuilder.server.luau
    
    Demonstrates how to preprocess a raw text corpus before training.
    
    The core problem with naive corpus passing: RoLLM's trimTokensToContext
    silently discards everything past maxSeqLen characters in any single
    corpus entry. A 500-line block of text becomes just the first ~32 chars.
    
    This example shows:
      1. Splitting multiline text into individual training samples
      2. Filtering out very short lines (noise)
      3. Normalising case and stripping extra whitespace
      4. Deduplicating near-identical lines
      5. Printing a corpus quality report before training
      
    Use buildCorpus() in your own projects to preprocess any raw text
    before passing it to trainModel.
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RoLLM = require(ReplicatedStorage.RoLLM)

local MIN_LINE_LENGTH = 8
local MAX_LINE_LENGTH = 60

local function buildCorpus(rawEntries: { string }): { string }
	local seen: { [string]: boolean } = {}
	local result: { string } = {}

	for _, entry in ipairs(rawEntries) do
		local lines: { string }

		if #entry > MAX_LINE_LENGTH then
			lines = string.split(entry, "\n")
		else
			lines = { entry }
		end

		for _, line in ipairs(lines) do
			local clean = line:match("^%s*(.-)%s*$") or line
			clean = clean:lower()
			clean = clean:gsub("%s+", " ")

			if #clean >= MIN_LINE_LENGTH and #clean <= MAX_LINE_LENGTH and not seen[clean] then
				seen[clean] = true
				table.insert(result, clean)
			end
		end
	end

	return result
end

local function corpusReport(raw: { string }, processed: { string })
	local rawChars = 0
	for _, s in ipairs(raw) do rawChars += #s end

	local processedChars = 0
	local shortest = math.huge
	local longest = 0
	for _, s in ipairs(processed) do
		processedChars += #s
		if #s < shortest then shortest = #s end
		if #s > longest then longest = #s end
	end

	print(string.rep("=", 50))
	print("Corpus report")
	print(string.rep("=", 50))
	print(string.format("  Raw entries      : %d", #raw))
	print(string.format("  Raw characters   : %d", rawChars))
	print(string.format("  Processed lines  : %d", #processed))
	print(string.format("  Avg length       : %.1f chars", processedChars / math.max(#processed, 1)))
	print(string.format("  Shortest line    : %d chars", shortest))
	print(string.format("  Longest line     : %d chars", longest))
	print(string.rep("=", 50))
end

local rawCorpus = {
	"you can move using wasd or the arrow keys.",
	"press space to jump.",
	"the round ends when the timer hits zero.",
	[[
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd.
	]],
	[[
All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances;
And one man in his time plays many parts,
His acts being seven ages.
	]],
	"the leaderboard shows the top ten players.",
	"gems respawn every thirty seconds.",
	"you lose points if you fall off the map.",
}

local trainingData = buildCorpus(rawCorpus)
corpusReport(rawCorpus, trainingData)

local llm = RoLLM.new(trainingData, {
	dModel = 32,
	numHeads = 4,
	dFF = 64,
	numLayers = 2,
	maxSeqLen = 32,
	tokenizerMode = "char",
})

print(string.format("Model: %d params, vocab: %d chars",
	llm:getParameterCount(), llm._tokenizer:getVocabSize()))

local history = llm:trainModel(trainingData, 15, 0.01, {
	batchSize = 8,
	yieldEverySamples = 1,
	onEpochComplete = function(epoch, total, avgLoss, elapsed)
		local eta = if epoch < total then (elapsed / epoch) * (total - epoch) else 0
		print(string.format("  Epoch %2d/%d | Loss: %.4f | ETA: %.0fs", epoch, total, avgLoss, eta))
	end,
})

print(string.format("\nDone. Final loss: %.4f", history[#history].averageLoss))

print("\n--- Samples ---")
for _, seed in ipairs({ "you can", "the lea", "to be, " }) do
	print(string.format("  %q -> %q", seed, llm:generateTemperature(seed, 40, 0.6)))
end
