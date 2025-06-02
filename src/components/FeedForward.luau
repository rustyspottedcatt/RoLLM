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

local function relu(mat: Matrix): Matrix
	local out = LinearAlgebra:zeros(#mat, #mat[1])
	for i = 1, #mat do
		for j = 1, #mat[i] do
			out[i][j] = math.max(0, mat[i][j])
		end
	end
	return out
end

local function reluBackward(x: Matrix, gradOutput: Matrix): Matrix
    local gradInput: Matrix = LinearAlgebra:zeros(#x, #x[1])
    for i = 1, #x do
        for j = 1, #x[i] do
            gradInput[i][j] = x[i][j] > 0 and gradOutput[i][j] or 0
        end
    end
    return gradInput
end

function FeedForward.new(dModel: number, dFF: number): FeedForwardClass
    local self = setmetatable({}, FeedForward)
    self.dModel = dModel
    self.dFF = dFF

    self.W1 = LinearAlgebra:randomMatrix(dModel, dFF, 0.01)
    self.b1 = {}
    for j = 1, dFF do
        self.b1[j] = 0
    end

    self.W2 = LinearAlgebra:randomMatrix(dFF, dModel, 0.01)
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
    self.cache.reluOutput = relu(hidden)

    -- hiddenW2 + b2
    local out = LinearAlgebra:mm(self.cache.reluOutput, self.W2)
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
                gradW2[k][j] += self.cache.reluOutput[i][k] * gradOutput[i][j]
            end
            gradB2[j] += gradOutput[i][j]
        end
    end

    -- grads w.r.t hidden (ReLU output)
    local gradHidden = LinearAlgebra:mm(gradOutput, LinearAlgebra:transpose(self.W2))

    -- grads through ReLU
    gradHidden = reluBackward(self.cache.hidden, gradHidden)

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
