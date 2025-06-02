--[[
    TransformerBlock Module

    @module TransformerBlock
    @type TransformerBlockClass
]]

local Types = require(script.Parent.Parent.lib.types)
local MultiHeadAttention = require(script.Parent.MultiHeadAttention)
local FeedForward = require(script.Parent.FeedForward)
local LayerNorm = require(script.Parent.LayerNorm)

export type Matrix = Types.Matrix

export type TransformerBlockClass = {
	new: (dModel: number, numHeads: number, dFF: number) -> TransformerBlockClass,
	dModel: number,
	mha: MultiHeadAttention.MultiHeadAttentionClass,
	ff: FeedForward.FeedForwardClass,
	norm1: LayerNorm.LayerNormClass,
	norm2: LayerNorm.LayerNormClass,
	forward: (self: TransformerBlockClass, x: Matrix, mask: Matrix?) -> Matrix,
	backward: (self: TransformerBlockClass, gradOutput: Matrix) -> Matrix,
	getParameters: (
		self: TransformerBlockClass
	) -> {
		mha: MultiHeadAttention.MultiHeadAttentionClass,
		ff: FeedForward.FeedForwardClass,
		norm1: LayerNorm.LayerNormClass,
		norm2: LayerNorm.LayerNormClass,
	},
	setParameters: (
		self: TransformerBlockClass,
		params: {
			mha: MultiHeadAttention.MultiHeadAttentionClass,
			ff: FeedForward.FeedForwardClass,
			norm1: LayerNorm.LayerNormClass,
			norm2: LayerNorm.LayerNormClass,
		}
	) -> nil,
}

local TransformerBlock = {}
TransformerBlock.__index = TransformerBlock

function TransformerBlock.new(dModel: number, numHeads: number, dFF: number): TransformerBlockClass
	local self = setmetatable({}, TransformerBlock)
	self.dModel = dModel
	self.mha = MultiHeadAttention.new(dModel, numHeads)
	self.ff = FeedForward.new(dModel, dFF)
	self.norm1 = LayerNorm.new(dModel)
	self.norm2 = LayerNorm.new(dModel)
	return self
end
function TransformerBlock:forward(x: Matrix, mask: Matrix?): Matrix
	local attnOut = self.mha:forward(x, mask)

	local x2_input = {}
	for i = 1, #x do
		x2_input[i] = {}
		for j = 1, self.dModel do
			x2_input[i][j] = x[i][j] + attnOut[i][j]
		end
	end
	local x2 = self.norm1:forward(x2_input)

	local ffOut = self.ff:forward(x2)

	local x3_input = {}
	for i = 1, #x2 do
		x3_input[i] = {}
		for j = 1, self.dModel do
			x3_input[i][j] = x2[i][j] + ffOut[i][j]
		end
	end
	local x3 = self.norm2:forward(x3_input)

	self.cache = {
		x = x,
		attnOut = attnOut,
		x2_input = x2_input,
		x2 = x2,
		ffOut = ffOut,
		x3_input = x3_input,
		x3 = x3,
	}

	return x3
end

function TransformerBlock:backward(gradOutput: Matrix): Matrix
	local grad_of_x3_input = self.norm2:backward(self.cache.x3_input, gradOutput)

	local gradX2_fromAdd2 = grad_of_x3_input -- goes into the FF block
	local gradFFOut_fromAdd2 = grad_of_x3_input -- goes back into x2

	local grad_of_x2_fromFF = self.ff:backward(gradX2_fromAdd2)

	local gradX2 = {}
	for i = 1, #grad_of_x2_fromFF do
		gradX2[i] = {}
		for j = 1, self.dModel do
			gradX2[i][j] = grad_of_x2_fromFF[i][j] + gradFFOut_fromAdd2[i][j]
		end
	end

	local grad_of_x2_input = self.norm1:backward(self.cache.x2_input, gradX2)

	local gradX_fromAdd1 = grad_of_x2_input -- goes into x
	local gradAttnFromAdd1 = grad_of_x2_input -- goes into attention output

	local gradX_fromMHA = self.mha:backward(gradAttnFromAdd1)

	local gradX = {}
	for i = 1, #gradX_fromMHA do
		gradX[i] = {}
		for j = 1, self.dModel do
			gradX[i][j] = gradX_fromAdd1[i][j] + gradX_fromMHA[i][j]
		end
	end

	return gradX
end

function TransformerBlock:getParameters(): {
	mha: MultiHeadAttention.MultiHeadAttentionClass,
	ff: FeedForward.FeedForwardClass,
	norm1: LayerNorm.LayerNormClass,
	norm2: LayerNorm.LayerNormClass,
}
	return {
		mha = self.mha,
		ff = self.ff,
		norm1 = self.norm1,
		norm2 = self.norm2,
	}
end

function TransformerBlock:setParameters(params: {
	mha: MultiHeadAttention.MultiHeadAttentionClass,
	ff: FeedForward.FeedForwardClass,
	norm1: LayerNorm.LayerNormClass,
	norm2: LayerNorm.LayerNormClass,
})
	self.mha = params.mha
	self.ff = params.ff
	self.norm1 = params.norm1
	self.norm2 = params.norm2
end

return TransformerBlock
