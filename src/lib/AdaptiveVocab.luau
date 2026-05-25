--[[
    AdaptiveVocab Module

    Wraps any base tokenizer (e.g. BPETokenizer with the full GPT-2 50K vocab) and
    builds a compact dense vocabulary from just the tokens that appear in the training
    corpus.  The model's embedding and output head operate over dense IDs 1..N
    instead of the full 50K, making both training and inference tractable in pure Luau.

    Usage:
        local av = AdaptiveVocab.new(bpeTokenizer, 2048)
        av:buildFromCorpus(corpusSentences)   -- one-time scan
        local denseIds = av:encode("press space to jump")
        local text     = av:decode(denseIds)

    @module AdaptiveVocab
]]

local AdaptiveVocab = {}
AdaptiveVocab.__index = AdaptiveVocab

export type AdaptiveVocabClass = {
	new: (baseTokenizer: any, maxVocabSize: number?) -> AdaptiveVocabClass,
	buildFromCorpus: (self: AdaptiveVocabClass, corpus: { string }) -> (),
	getVocabSize: (self: AdaptiveVocabClass) -> number,
	encode: (self: AdaptiveVocabClass, text: string) -> { number },
	decode: (self: AdaptiveVocabClass, denseIds: { number }) -> string,
	unkDenseId: number,
	vocabSize: number,
}

function AdaptiveVocab.new(baseTokenizer: any, maxVocabSize: number?): AdaptiveVocabClass
	local self = setmetatable({}, AdaptiveVocab)
	self.baseTokenizer  = baseTokenizer
	self.maxVocabSize   = maxVocabSize or 2048
	self.gptToDense     = {} :: { [number]: number }
	self.denseToGpt     = {} :: { [number]: number }
	self.vocabSize      = 0
	self.unkDenseId     = 1
	return self
end

-- Scan corpus sentences, count token frequencies, keep top maxVocabSize.
-- Dense ID 1 is reserved for UNK; real tokens start at 2.
function AdaptiveVocab:buildFromCorpus(corpus: { string })
	local freqs: { [number]: number } = {}
	for _, sentence in ipairs(corpus) do
		local tokens = self.baseTokenizer:textToTokens(sentence)
		for _, id in ipairs(tokens) do
			freqs[id] = (freqs[id] or 0) + 1
		end
	end

	local items: { { id: number, count: number } } = {}
	for id, count in pairs(freqs) do
		table.insert(items, { id = id, count = count })
	end
	table.sort(items, function(a, b) return a.count > b.count end)

	self.gptToDense = {}
	self.denseToGpt = {}
	self.unkDenseId = 1

	local denseId = 2
	local limit   = math.min(#items, self.maxVocabSize - 1)
	for i = 1, limit do
		local gptId = items[i].id
		self.gptToDense[gptId] = denseId
		self.denseToGpt[denseId] = gptId
		denseId += 1
	end

	self.vocabSize = denseId - 1

	print(string.format(
		"[AdaptiveVocab] %d unique base tokens in corpus → kept top %d → dense vocab size %d",
		#items, limit, self.vocabSize
	))
end

function AdaptiveVocab:getVocabSize(): number
	return self.vocabSize
end

-- Encode a string to dense IDs.  Tokens not in the active vocab map to UNK (1).
function AdaptiveVocab:encode(text: string): { number }
	local rawIds   = self.baseTokenizer:textToTokens(text)
	local denseIds: { number } = {}
	for _, gptId in ipairs(rawIds) do
		local d = self.gptToDense[gptId] or self.unkDenseId
		table.insert(denseIds, d)
	end
	return denseIds
end

-- Decode dense IDs back to a string.  UNK tokens are silently dropped.
function AdaptiveVocab:decode(denseIds: { number }): string
	local gptIds: { number } = {}
	for _, d in ipairs(denseIds) do
		local gptId = self.denseToGpt[d]
		if gptId then
			table.insert(gptIds, gptId)
		end
	end
	return self.baseTokenizer:tokensToText(gptIds)
end

return AdaptiveVocab
