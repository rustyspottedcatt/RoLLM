--[[
    FeedForward Module

    @module FeedForward
    @type FeedForwardClass
]]

local Types = require(script.Parent.Parent.lib.types)
local LinearAlgebraModule = require(script.Parent.LinearAlgebra)

export type Matrix = Types.Matrix
export type Vector = Types.Vector

export type FeedForwardClass = {
	new: (dModel: number, dFF: number) -> FeedForwardClass,
	dModel: number,
	dFF: number,
	W1: Matrix,
	b1: Vector,
	W2: Matrix,
	b2: Vector,
	forward: (self: FeedForwardClass, x: Matrix) -> Matrix,
	backward: (self: FeedForwardClass, gradOutput: Matrix) -> Matrix,
	getParameters: (self: FeedForwardClass) -> { W1: Matrix, b1: Vector, W2: Matrix, b2: Vector },
	setParameters: (self: FeedForwardClass, params: { W1: Matrix, b1: Vector, W2: Matrix, b2: Vector }) -> boolean,
}

local FeedForward = {}
FeedForward.__index = FeedForward

local LinearAlgebra = LinearAlgebraModule.new()

local SQRT_2_OVER_PI = math.sqrt(2 / math.pi)  -- ≈ 0.7978845608
local GELU_COEFF     = 0.044715

-- GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
local function gelu(mat: Matrix): Matrix
	local out = LinearAlgebra:zeros(#mat, #mat[1])
	for i = 1, #mat do
		for j = 1, #mat[i] do
			local x = mat[i][j]
			local inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)
			out[i][j] = 0.5 * x * (1 + math.tanh(inner))
		end
	end
	return out
end

-- Exact GELU backward.
local function geluBackward(x: Matrix, gradOutput: Matrix): Matrix
	local gradInput: Matrix = LinearAlgebra:zeros(#x, #x[1])
	for i = 1, #x do
		for j = 1, #x[i] do
			local xv    = x[i][j]
			local inner = SQRT_2_OVER_PI * (xv + GELU_COEFF * xv * xv * xv)
			local tanh_inner = math.tanh(inner)
			local sech2      = 1 - tanh_inner * tanh_inner
			local dInner_dx  = SQRT_2_OVER_PI * (1 + 3 * GELU_COEFF * xv * xv)
			local dgelu_dx   = 0.5 * (1 + tanh_inner) + 0.5 * xv * sech2 * dInner_dx
			gradInput[i][j]  = dgelu_dx * gradOutput[i][j]
		end
	end
	return gradInput
end

function FeedForward.new(dModel: number, dFF: number): FeedForwardClass
	local self = setmetatable({}, FeedForward)
	self.dModel = dModel
	self.dFF = dFF

	self.W1 = LinearAlgebra:xavierMatrix(dModel, dFF)
	self.b1 = {}
	for j = 1, dFF do
		self.b1[j] = 0
	end

	self.W2 = LinearAlgebra:xavierMatrix(dFF, dModel)
	self.b2 = {}
	for j = 1, dModel do
		self.b2[j] = 0
	end

	self.cache = {}

	return self
end

function FeedForward:forward(x: Matrix): Matrix
	assert(#x[1] == self.dModel, "FeedForward: input does not match dModel")

	local seqLen = #x

	-- Compute xW1 + b1
	local hidden = LinearAlgebra:mm(x, self.W1)
	for i = 1, seqLen do
		for j = 1, self.dFF do
			hidden[i][j] += self.b1[j]
		end
	end

	self.cache.x = x
	self.cache.hidden = hidden
	self.cache.geluOutput = gelu(hidden)

	-- hiddenW2 + b2
	local out = LinearAlgebra:mm(self.cache.geluOutput, self.W2)
	for i = 1, seqLen do
		for j = 1, self.dModel do
			out[i][j] += self.b2[j]
		end
	end

	self.cache.output = out

	return out
end

function FeedForward:backward(gradOutput: Matrix): Matrix
	assert(#gradOutput[1] == self.dModel, "FeedForward: gradOutput shape mismatch")

	local seqLen = #gradOutput

	-- grads w.r.t W2 and b2
	local gradW2 = LinearAlgebra:zeros(self.dFF, self.dModel)
	local gradB2 = {}
	for j = 1, self.dModel do
		gradB2[j] = 0
	end

	for i = 1, seqLen do
		for j = 1, self.dModel do
			for k = 1, self.dFF do
				gradW2[k][j] += self.cache.geluOutput[i][k] * gradOutput[i][j]
			end
			gradB2[j] += gradOutput[i][j]
		end
	end

	-- grads w.r.t hidden (GELU output)
	local gradHidden = LinearAlgebra:mm(gradOutput, LinearAlgebra:transpose(self.W2))

	-- grads through GELU
	gradHidden = geluBackward(self.cache.hidden, gradHidden)

	-- grads w.r.t W1 and b1
	local gradW1 = LinearAlgebra:zeros(self.dModel, self.dFF)
	local gradB1 = {}
	for j = 1, self.dFF do
		gradB1[j] = 0
	end

	for i = 1, seqLen do
		for j = 1, self.dFF do
			for k = 1, self.dModel do
				gradW1[k][j] += self.cache.x[i][k] * gradHidden[i][j]
			end
			gradB1[j] += gradHidden[i][j]
		end
	end

	self.gradW2 = gradW2
	self.gradB2 = gradB2
	self.gradW1 = gradW1
	self.gradB1 = gradB1

	local gradX = LinearAlgebra:mm(gradHidden, LinearAlgebra:transpose(self.W1))

	return gradX
end

function FeedForward:getParameters(): { W1: Matrix, b1: Vector, W2: Matrix, b2: Vector }
	return {
		W1 = self.W1,
		b1 = self.b1,
		W2 = self.W2,
		b2 = self.b2,
	}
end

function FeedForward:setParameters(params: { W1: Matrix, b1: Vector, W2: Matrix, b2: Vector }): boolean
	self.W1 = params.W1
	self.b1 = params.b1
	self.W2 = params.W2
	self.b2 = params.b2

	return true
end

return FeedForward
