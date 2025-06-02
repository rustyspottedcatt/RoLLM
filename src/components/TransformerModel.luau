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
	numLayers: number,
	blocks: { TransformerBlock.TransformerBlockClass },
	embedding: Embedding.EmbeddingClass,
	finalProj: Matrix,
	gradFinalProj: Matrix?,
	forward: (self: TransformerModelClass, tokens: { number }) -> Matrix,
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

	self.vocabSize = config.vocabSize or 0
	self.dModel = config.dModel
	self.numLayers = config.numLayers or 2
	self.embedding = Embedding.new(self.vocabSize, self.dModel, config.maxSeqLen)
	self.blocks = {}

	for i = 1, self.numLayers do
		self.blocks[i] = TransformerBlock.new(config.dModel, config.numHeads, config.dFF)
	end

	self.finalProj = LinearAlgebra:randomMatrix(self.dModel, self.vocabSize, 0.01)
	self.gradFinalProj = nil
	self.cache = {}

	return self
end

function TransformerModel:forward(tokens: { number }): Matrix
	local seqLen = #tokens

	local positions: { number } = {}
	for i = 1, seqLen do
		positions[i] = i
	end

	local x = self.embedding:forward(tokens, positions)

	local mask = LinearAlgebra:zeros(seqLen, seqLen)
	for i = 1, seqLen do
		for j = 1, seqLen do
			if j > i then
				mask[i][j] = -1e9
			end
		end
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
	local lastHidden = self.cache.lastHidden
	local gradFinalProj = LinearAlgebra:mm(LinearAlgebra:transpose(lastHidden), gradOutput)
	self.gradFinalProj = gradFinalProj 

	local gradLastHidden = LinearAlgebra:mm(gradOutput, LinearAlgebra:transpose(self.finalProj))

	local gradBlockOut = gradLastHidden
	for i = self.numLayers, 1, -1 do
		gradBlockOut = self.blocks[i]:backward(gradBlockOut)
	end

	local embeddingBack = self.embedding:backward(gradBlockOut)
	local gradEmbeddingWeights = embeddingBack.weightsGrad

	return {
		gradEmbedding = gradBlockOut,
		gradFinalProj = gradFinalProj,
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
