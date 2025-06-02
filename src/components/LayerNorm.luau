--[[
    LayerNorm Module

    @module LayerNorm
    @type LayerNormClass
]]

local Types = require(script.Parent.Parent.lib.types)

export type Matrix = Types.Matrix

export type LayerNormClass = {
	new: (dModel: number, eps: number?) -> LayerNormClass,
	dModel: number,
	eps: number,
	gamma: { number },
	beta: { number },
	forward: (self: LayerNormClass, x: Matrix) -> Matrix,
	backward: (self: LayerNormClass, x: Matrix, gradOutput: Matrix) -> Matrix,
	getParameters: (self: LayerNormClass) -> { gamma: { number }, beta: { number } },
	setParameters: (self: LayerNormClass, params: { gamma: { number }, beta: { number } }) -> nil,
}

local LayerNorm = {}
LayerNorm.__index = LayerNorm

function LayerNorm.new(dModel: number, eps: number?): LayerNormClass
	local self = setmetatable({}, LayerNorm)
	self.dModel = dModel
	self.eps = eps or 1e-5

	self.gamma = {}
	self.beta = {}
	for i = 1, dModel do
		self.gamma[i] = 1
		self.beta[i] = 0
	end

	self.cache = {}

	return self
end

function LayerNorm:forward(x: Matrix): Matrix
	local seqLen = #x
	local out: Matrix = {}

	local mean = {}
	local var = {}
	for i = 1, seqLen do
		mean[i] = 0
		for j = 1, self.dModel do
			mean[i] += x[i][j]
		end
		mean[i] /= self.dModel

		var[i] = 0
		for j = 1, self.dModel do
			local diff = x[i][j] - mean[i]
			var[i] += diff * diff
		end
		var[i] /= self.dModel
	end

	local normalized = {}

	for i = 1, seqLen do
		out[i] = {}
		local normRow = {}
		for j = 1, self.dModel do
			local norm = (x[i][j] - mean[i]) / math.sqrt(var[i] + self.eps)
			normRow[j] = norm
			out[i][j] = norm * self.gamma[j] + self.beta[j]
		end
		normalized[i] = normRow
	end

	self.cache = {
		x = x,
		mean = mean,
		var = var,
		normalized = normalized,
	}

	return out
end

function LayerNorm:backward(x: Matrix, gradOutput: Matrix): Matrix
	local seqLen = #x
	local gradInput: Matrix = {}
	local gradGamma: { number } = {}
	local gradBeta: { number } = {}

	local gamma = self.gamma
	local eps = self.eps
	local mean = self.cache.mean
	local var = self.cache.var
	local normalized = self.cache.normalized

	-- gradients
	for j = 1, self.dModel do
		gradGamma[j] = 0
		gradBeta[j] = 0
	end

	-- grads w.r.t gamma and beta
	for i = 1, seqLen do
		for j = 1, self.dModel do
			gradGamma[j] += gradOutput[i][j] * normalized[i][j]
			gradBeta[j] += gradOutput[i][j]
		end
	end

	-- grads w.r.t input x
	for i = 1, seqLen do
		gradInput[i] = {}

		local std = math.sqrt(var[i] + eps)
		local x_hat = normalized[i]

		local dxhat: { number } = {}
		local dvar = 0
		local dmean = 0

		for j = 1, self.dModel do
			dxhat[j] = gradOutput[i][j] * gamma[j]
		end

		-- dvar = sum(dxhat * (x - mean) * -0.5 * std^-3)
		for j = 1, self.dModel do
			local xmu = x[i][j] - mean[i]
			dvar += dxhat[j] * xmu * -0.5 / (std ^ 3)
		end

		-- dmean = sum(dxhat * -1 / std) + dvar * sum(-2 * (x - mean)) / dModel
		for j = 1, self.dModel do
			dmean += dxhat[j] * -1 / std
		end
		for j = 1, self.dModel do
			dmean += dvar * -2 * (x[i][j] - mean[i]) / self.dModel
		end

		--  output
		for j = 1, self.dModel do
			local xmu = x[i][j] - mean[i]
			gradInput[i][j] = dxhat[j] / std + dvar * 2 * xmu / self.dModel + dmean / self.dModel
		end
	end

	self.gradGamma = gradGamma
	self.gradBeta = gradBeta

	return gradInput
end

function LayerNorm:getParameters(): { gamma: { number }, beta: { number } }
	return {
		gamma = self.gamma,
		beta = self.beta,
	}
end

function LayerNorm:setParameters(params: { gamma: { number }, beta: { number } })
	for j, value in ipairs(params.gamma) do
		self.gamma[j] = value
	end
	for j, value in ipairs(params.beta) do
		self.beta[j] = value
	end
end

return LayerNorm
