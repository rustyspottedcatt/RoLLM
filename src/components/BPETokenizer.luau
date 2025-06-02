--[[
    BPETokenizer Module

    @module BPETokenizer
    @type BPETokenizerClass
]]

local HttpService = game:GetService("HttpService")

local BPETokenizer = {}
BPETokenizer.__index = BPETokenizer

export type BPETokenizerClass = {
	new: () -> BPETokenizerClass,
	loadExternalVocab: (self: BPETokenizerClass, url: string) -> (),
	getVocabSize: (self: BPETokenizerClass) -> number,
	splitBPE: (self: BPETokenizerClass, word: string) -> { string },
	textToTokens: (self: BPETokenizerClass, text: string) -> { number },
	tokensToText: (self: BPETokenizerClass, tokens: { number }) -> string,
}

function BPETokenizer.new(): BPETokenizerClass
	local self = setmetatable({}, BPETokenizer)
	self.vocab = {}
	self.ivocab = {}
	self.vocabSize = 0
	self.unknownToken = "<UNK>"
	return self
end

function BPETokenizer:loadExternalVocab(url: string): ()
	local success, result = pcall(function()
		return HttpService:GetAsync(url)
	end)

	if not success then
		error("Failed to fetch external vocab from URL: " .. tostring(result))
	end

	local successDecode, decodeResult = pcall(function()
		return HttpService:JSONDecode(result)
	end)
	if not successDecode then
		error("Failed to decode external vocab JSON.")
	end

	for token, tokenId in pairs(decodeResult) do
		self.vocab[token] = tokenId
		self.ivocab[tokenId] = token
	end

	if not self.vocab[self.unknownToken] then
		local maxID = 0
		for _, tid in pairs(self.vocab) do
			if tid > maxID then
				maxID = tid
			end
		end
		local newID = maxID + 1
		self.vocab[self.unknownToken] = newID
		self.ivocab[newID] = self.unknownToken
	end

	local highest = 0
	for _, tid in pairs(self.vocab) do
		if tid > highest then
			highest = tid
		end
	end
	self.vocabSize = highest
end

function BPETokenizer:getVocabSize(): number
	return self.vocabSize
end

function BPETokenizer:splitBPE(word: string): { string }
	local tokens: { string } = {}
	local i = 1
	while i <= #word do
		local matched = false
		for j = #word, i, -1 do
			local sub = word:sub(i, j)
			if self.vocab[sub] then
				table.insert(tokens, sub)
				i = j + 1
				matched = true
				break
			end
		end
		if not matched then
			table.insert(tokens, self.unknownToken)
			break
		end
	end
	return tokens
end

function BPETokenizer:textToTokens(text: string): { number }
	local tokens: { number } = {}
	for word in text:gmatch("%S+") do
		local prefixedWord = " " .. word
		local wordTokens = self:splitBPE(prefixedWord)
		for _, tok in ipairs(wordTokens) do
			local id = self.vocab[tok]
			if not id then
				id = self.vocab[self.unknownToken]
			end
			table.insert(tokens, id)
		end
	end
	return tokens
end

function BPETokenizer:tokensToText(tokens: { number }): string
	local words: { string } = {}
	for _, id in ipairs(tokens) do
		local tk = self.ivocab[id] or self.unknownToken
		table.insert(words, tk)
	end
	return table.concat(words, "")
end

return BPETokenizer
