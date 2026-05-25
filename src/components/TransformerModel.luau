--[[
    TransformerModel Module

    @module TransformerModel
    @type TransformerModelClass
]]

local Types = require(script.Parent.Parent.lib.types)
local LinearAlgebraModule = require(script.Parent.LinearAlgebra)
local Embedding = require(script.Parent.Embedding)
local TransformerBlock = require(script.Parent.TransformerBlock)

export type Matrix = Types.Matrix
export type TransformerConfig = Types.TransformerConfig

export type TransformerModelClass = {
	new: (config: TransformerConfig) -> TransformerModelClass,
	vocabSize: number,
	dModel: number,
	maxSeqLen: number,
	numLayers: number,
	weightTied: boolean,
	blocks: { TransformerBlock.TransformerBlockClass },
	embedding: Embedding.EmbeddingClass,
	finalProj: Matrix,
	gradFinalProj: Matrix?,
	forward: (self: TransformerModelClass, tokens: { number }) -> Matrix,
	forwardCached: (self: TransformerModelClass, tokens: { number }, startPos: number?) -> Matrix,
	clearKVCache: (self: TransformerModelClass) -> (),
	backward: (
		self: TransformerModelClass,
		gradOutput: Matrix
	) -> {
		gradEmbedding: Matrix,
		gradFinalProj: Matrix,
		gradEmbeddingWeights: Matrix,
	},
	predictNextToken: (self: TransformerModelClass, tokens: { number }) -> number,
	predictNextTokenTemperature: (self: TransformerModelClass, tokens: { number }, temperature: number) -> number,
	predictNextTokenTopK: (self: TransformerModelClass, tokens: { number }, k: number, temperature: number?) -> number,
	predictNextTokenNucleus: (self: TransformerModelClass, tokens: { number }, p: number, temperature: number?) -> number,
	getParameters: (
		self: TransformerModelClass
	) -> { embedding: Embedding.EmbeddingClass, blocks: { TransformerBlock.TransformerBlockClass }, finalProj: Matrix },
	setParameters: (
		self: TransformerModelClass,
		params: {
			embedding: Embedding.EmbeddingClass,
			blocks: { TransformerBlock.TransformerBlockClass },
			finalProj: Matrix,
		}
	) -> nil,
}

local TransformerModel = {}
TransformerModel.__index = TransformerModel

local LinearAlgebra = LinearAlgebraModule.new()

function TransformerModel.new(config: TransformerConfig): TransformerModelClass
	local self = setmetatable({}, TransformerModel)

	self.vocabSize  = config.vocabSize or 0
	self.dModel     = config.dModel
	self.maxSeqLen  = config.maxSeqLen
	self.numLayers  = config.numLayers or 2
	self.weightTied = config.weightTying == true
	self.embedding  = Embedding.new(self.vocabSize, self.dModel, config.maxSeqLen)
	self.blocks     = {}

	for i = 1, self.numLayers do
		self.blocks[i] = TransformerBlock.new(config.dModel, config.numHeads, config.dFF)
	end

	if self.weightTied then
		-- finalProj = embedding.weightsᵀ  (vocabSize×dModel → dModel×vocabSize)
		-- They share the same underlying matrix; finalProj is rebuilt each forward pass.
		self.finalProj = LinearAlgebra:transpose(self.embedding.weights)
	else
		self.finalProj = LinearAlgebra:xavierMatrix(self.dModel, self.vocabSize)
	end

	self.gradFinalProj = nil
	self.cache         = {}
	self.positionCache = {}
	self.maskCache     = {}

	return self
end

function TransformerModel:forward(tokens: { number }): Matrix
	local seqLen = #tokens
	assert(seqLen > 0, "TransformerModel: tokens cannot be empty")
	assert(seqLen <= self.maxSeqLen, "TransformerModel: sequence length exceeds maxSeqLen")
	if self.weightTied then
		self.finalProj = LinearAlgebra:transpose(self.embedding.weights)
	end

	local positions = self.positionCache[seqLen]
	if not positions then
		positions = {}
		for i = 1, seqLen do
			positions[i] = i
		end
		self.positionCache[seqLen] = positions
	end

	local x = self.embedding:forward(tokens, positions)

	local mask = self.maskCache[seqLen]
	if not mask then
		mask = LinearAlgebra:zeros(seqLen, seqLen)
		for i = 1, seqLen do
			for j = i + 1, seqLen do
				mask[i][j] = -1e9
			end
		end
		self.maskCache[seqLen] = mask
	end

	for _, block in ipairs(self.blocks) do
		x = block:forward(x, mask)
	end

	local logits = LinearAlgebra:mm(x, self.finalProj)

	self.cache = {
		tokens = tokens,
		positions = positions,
		lastHidden = x,
		logits = logits,
	}

	return logits
end

function TransformerModel:backward(
	gradOutput: Matrix
): { gradEmbedding: Matrix, gradFinalProj: Matrix, gradEmbeddingWeights: Matrix }
	local lastHidden    = self.cache.lastHidden
	local gradFinalProj = LinearAlgebra:mm(LinearAlgebra:transpose(lastHidden), gradOutput)
	self.gradFinalProj  = gradFinalProj

	local gradLastHidden = LinearAlgebra:mm(gradOutput, LinearAlgebra:transpose(self.finalProj))

	local gradBlockOut = gradLastHidden
	for i = self.numLayers, 1, -1 do
		gradBlockOut = self.blocks[i]:backward(gradBlockOut)
	end

	local embeddingBack        = self.embedding:backward(gradBlockOut)
	local gradEmbeddingWeights = embeddingBack.weightsGrad

	if self.weightTied then
		-- Accumulate gradFinalProjᵀ into embedding weights grad
		local gradFinalProjT = LinearAlgebra:transpose(gradFinalProj)
		for v = 1, self.vocabSize do
			for d = 1, self.dModel do
				gradEmbeddingWeights[v][d] += gradFinalProjT[v][d]
			end
		end
	end

	return {
		gradEmbedding        = gradBlockOut,
		gradFinalProj        = gradFinalProj,
		gradEmbeddingWeights = gradEmbeddingWeights,
	}
end

function TransformerModel:predictNextToken(tokens: { number }): number
	local logits = self:forward(tokens)
	local seqLen = #logits
	local lastLogits = logits[seqLen]

	local maxVal = -math.huge
	local maxIndex = 1
	for i, val in ipairs(lastLogits) do
		if val > maxVal then
			maxVal = val
			maxIndex = i
		end
	end
	return maxIndex
end

-- Inference pass with KV cache.  startPos is the 1-based position offset for the
-- first token in `tokens` (default 1 for a fresh sequence).
function TransformerModel:forwardCached(tokens: { number }, startPos: number?): Matrix
	local seqLen = #tokens
	assert(seqLen > 0, "TransformerModel: tokens cannot be empty")

	local offset = (startPos or 1) - 1
	local positions: { number } = {}
	for i = 1, seqLen do
		positions[i] = offset + i
	end

	if self.weightTied then
		self.finalProj = LinearAlgebra:transpose(self.embedding.weights)
	end

	local x = self.embedding:forward(tokens, positions)

	-- Build causal mask for initial prompt fill (when seqLen > 1)
	local mask: Matrix? = nil
	if seqLen > 1 then
		local cached = self.maskCache[seqLen]
		if cached then
			mask = cached
		else
			local newMask = LinearAlgebra:zeros(seqLen, seqLen)
			for i = 1, seqLen do
				for j = i + 1, seqLen do
					newMask[i][j] = -1e9
				end
			end
			self.maskCache[seqLen] = newMask
			mask = newMask
		end
	end

	for _, block in ipairs(self.blocks) do
		x = block:forwardCached(x, mask)
	end

	return LinearAlgebra:mm(x, self.finalProj)
end

function TransformerModel:clearKVCache()
	for _, block in ipairs(self.blocks) do
		block:clearKVCache()
	end
end

function TransformerModel:predictNextTokenTemperature(tokens: { number }, temperature: number): number
	local logits = self:forward(tokens)
	local seqLen = #logits
	local lastLogits = logits[seqLen]
	local vocabSize = #lastLogits

	if temperature < 1e-7 then
		local maxVal = -math.huge
		local maxIndex = 1
		for i, val in ipairs(lastLogits) do
			if val > maxVal then
				maxVal = val
				maxIndex = i
			end
		end
		return maxIndex
	end

	local scaled: { number } = {}
	local maxLogit = -math.huge
	for i = 1, vocabSize do
		scaled[i] = lastLogits[i] / temperature
		if scaled[i] > maxLogit then
			maxLogit = scaled[i]
		end
	end

	local sumExp = 0
	for i = 1, vocabSize do
		scaled[i] = math.exp(scaled[i] - maxLogit)
		sumExp += scaled[i]
	end
	for i = 1, vocabSize do
		scaled[i] = scaled[i] / sumExp
	end

	local randVal = math.random()
	local cumulative = 0
	for i = 1, vocabSize do
		cumulative += scaled[i]
		if randVal <= cumulative then
			return i
		end
	end

	return vocabSize
end

-- Top-k sampling: keep only the k highest-logit tokens, sample from their distribution.
function TransformerModel:predictNextTokenTopK(tokens: { number }, k: number, temperature: number?): number
	local logits    = self:forward(tokens)
	local seqLen    = #logits
	local lastLogits = logits[seqLen]
	local vocabSize  = #lastLogits
	local temp       = temperature or 1.0

	local pairs_list: { { i: number, v: number } } = {}
	for i = 1, vocabSize do
		table.insert(pairs_list, { i = i, v = lastLogits[i] })
	end
	table.sort(pairs_list, function(a, b) return a.v > b.v end)

	local topK = math.min(k, vocabSize)
	local scaled: { number } = {}
	local ids: { number }    = {}
	local maxLogit = pairs_list[1].v
	for rank = 1, topK do
		ids[rank]    = pairs_list[rank].i
		scaled[rank] = math.exp((pairs_list[rank].v - maxLogit) / math.max(temp, 1e-7))
	end

	local sumExp = 0
	for _, v in ipairs(scaled) do sumExp += v end

	local randVal    = math.random() * sumExp
	local cumulative = 0
	for rank = 1, topK do
		cumulative += scaled[rank]
		if randVal <= cumulative then return ids[rank] end
	end
	return ids[topK]
end

-- Nucleus (top-p) sampling: keep the smallest set of tokens whose cumulative
-- probability ≥ p, sample from their distribution.
function TransformerModel:predictNextTokenNucleus(tokens: { number }, p: number, temperature: number?): number
	local logits     = self:forward(tokens)
	local seqLen     = #logits
	local lastLogits  = logits[seqLen]
	local vocabSize   = #lastLogits
	local temp        = temperature or 1.0

	local pairs_list: { { i: number, v: number } } = {}
	for i = 1, vocabSize do
		table.insert(pairs_list, { i = i, v = lastLogits[i] })
	end
	table.sort(pairs_list, function(a, b) return a.v > b.v end)

	-- Softmax over sorted logits
	local maxLogit = pairs_list[1].v
	local probs: { number } = {}
	local sumExp = 0
	for _, entry in ipairs(pairs_list) do
		local e = math.exp((entry.v - maxLogit) / math.max(temp, 1e-7))
		table.insert(probs, e)
		sumExp += e
	end
	for i = 1, vocabSize do probs[i] = probs[i] / sumExp end

	-- Collect tokens until cumulative prob ≥ p
	local nucleus: { { i: number, prob: number } } = {}
	local cumProb = 0
	for rank = 1, vocabSize do
		table.insert(nucleus, { i = pairs_list[rank].i, prob = probs[rank] })
		cumProb += probs[rank]
		if cumProb >= p then break end
	end

	-- Re-normalise nucleus and sample
	local nucSum = 0
	for _, entry in ipairs(nucleus) do nucSum += entry.prob end
	local randVal    = math.random() * nucSum
	local cumulative = 0
	for _, entry in ipairs(nucleus) do
		cumulative += entry.prob
		if randVal <= cumulative then return entry.i end
	end
	return nucleus[#nucleus].i
end

function TransformerModel:getParameters(): {
	embedding: Embedding.EmbeddingClass,
	blocks: { TransformerBlock.TransformerBlockClass },
	finalProj: Matrix,
}
	return {
		embedding = self.embedding,
		blocks = self.blocks,
		finalProj = self.finalProj,
	}
end

function TransformerModel:setParameters(params: {
	embedding: Embedding.EmbeddingClass,
	blocks: { TransformerBlock.TransformerBlockClass },
	finalProj: Matrix,
})
	self.embedding = params.embedding
	self.blocks = params.blocks
	self.finalProj = params.finalProj
end

return TransformerModel
