-- Embedding.lua

--[[
    Embedding Module

    @module Embedding
    @type EmbeddingClass
]]

local Types = require(script.Parent.Parent.lib.types)
local LinearAlgebraModule = require(script.Parent.LinearAlgebra)

export type EmbeddingClass = {
	new: (vocabSize: number, dModel: number, maxSeqLen: number) -> EmbeddingClass,
	vocabSize: number,
	dModel: number,
	maxSeqLen: number,
	weights: { { number } },
	positions: { { number } }, 
	forward: (self: EmbeddingClass, tokens: { number }, positions: { number }) -> { { number } },
	backward: (
		self: EmbeddingClass,
		gradOutput: { { number } }
	) -> { positionsGrad: { { number } }, weightsGrad: { { number } } },
}

local Embedding = {}
Embedding.__index = Embedding

local LinearAlgebra = LinearAlgebraModule.new()

function Embedding.new(vocabSize: number, dModel: number, maxSeqLen: number): EmbeddingClass
	local self = setmetatable({}, Embedding)
	self.vocabSize = vocabSize
	self.dModel = dModel
	self.maxSeqLen = maxSeqLen
	self.weights = LinearAlgebra:randomMatrix(vocabSize, dModel, 0.01)
	self.positions = {}

	for pos = 1, maxSeqLen do
		local row = {}
		for i = 1, dModel do
			local angle = pos / (10000 ^ ((i - 1) / dModel))
			if i % 2 == 1 then
				row[i] = math.sin(angle)
			else
				row[i] = math.cos(angle)
			end
		end
		self.positions[pos] = row
	end

	self.cache = {
		tokens = nil,
		positions = nil,
		embeddedTokens = nil, -- (seqLen × dModel) sum of token‐embed + pos‐embed
	}

	return self
end

function Embedding:forward(tokens: { number }, positions: { number }): { { number } }
	local seqLen = #tokens
	local out: { { number } } = {}

	for i = 1, seqLen do
		local tokenID = tokens[i]
		local tokenEmbedRow = {}
		for j = 1, self.dModel do
			tokenEmbedRow[j] = self.weights[tokenID][j]
		end

		local posID = positions[i]
		local posRow = self.positions[posID]
		local sumRow = {}
		for j = 1, self.dModel do
			sumRow[j] = tokenEmbedRow[j] + posRow[j]
		end

		out[i] = sumRow
	end

	self.cache.tokens = { table.unpack(tokens) } -- copy of tokens
	self.cache.positions = { table.unpack(positions) } -- copy of positions
	self.cache.embeddedTokens = out -- (seqLen × dModel)

	return out
end

function Embedding:backward(gradOutput: { { number } }): { positionsGrad: { { number } }, weightsGrad: { { number } } }
	local tokens = self.cache.tokens -- { tokenID, ... }
	local positions = self.cache.positions -- { posID, ... }
	local seqLen = #tokens

	local positionsGrad: { { number } } = {}
	for i = 1, seqLen do
		positionsGrad[i] = {}
		for j = 1, self.dModel do
			positionsGrad[i][j] = gradOutput[i][j] 
		end
	end

	local weightsGrad: { { number } } = {}
	for v = 1, self.vocabSize do
		weightsGrad[v] = {}
		for j = 1, self.dModel do
			weightsGrad[v][j] = 0
		end
	end

	for i = 1, seqLen do
		local tokenID = tokens[i]
		for j = 1, self.dModel do
			weightsGrad[tokenID][j] += gradOutput[i][j]
		end
	end

	return {
		positionsGrad = positionsGrad,
		weightsGrad = weightsGrad,
	}
end

return Embedding
