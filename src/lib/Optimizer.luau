--[[
    Optimizer Module

    @module Optimizer
    @type OptimizerClass
]]

export type OptimizerClass = {
	new: (learningRate: number, beta1: number?, beta2: number?, epsilon: number?) -> OptimizerClass,
	learningRate: number,
	beta1: number,
	beta2: number,
	epsilon: number,
	m: { [string]: any },
	v: { [string]: any },
	t: number,
	updateParameters: (self: OptimizerClass, parameters: { any }, learningRate: number) -> number,
	clipGradNorm: (self: OptimizerClass, parameters: { [string]: { value: any, grad: any } }, maxNorm: number) -> number,
	getLR: (self: OptimizerClass, epoch: number, totalEpochs: number, baseLR: number, warmupEpochs: number?) -> number,
}

local Optimizer = {}
Optimizer.__index = Optimizer

function Optimizer.new(learningRate: number, beta1: number?, beta2: number?, epsilon: number?): OptimizerClass
	local self = setmetatable({}, Optimizer)
	self.learningRate = learningRate
	self.beta1 = beta1 or 0.9
	self.beta2 = beta2 or 0.999
	self.epsilon = epsilon or 1e-8
	self.m = {}
	self.v = {}
	self.t = 0
	return self
end

function Optimizer:updateParameters(parameters: { [string]: { value: any, grad: any } }, learningRate: number?): number
	self.t = self.t + 1
	local lr = learningRate or self.learningRate
	local count = 0

	for paramName, paramStruct in pairs(parameters) do
		local W = paramStruct.value
		local G = paramStruct.grad
		local firstValue = W[1]

		if type(firstValue) == "table" then
			if not self.m[paramName] then
				self.m[paramName] = {}
				self.v[paramName] = {}
				for i, row in ipairs(W) do
					self.m[paramName][i] = {}
					self.v[paramName][i] = {}
					for j, _ in ipairs(row) do
						self.m[paramName][i][j] = 0
						self.v[paramName][i][j] = 0
					end
				end
			end

			for i, row in ipairs(W) do
				for j, _ in ipairs(row) do
					local grad_ij = if G[i] then G[i][j] or 0 else 0

					local m_ij = self.m[paramName][i][j]
					m_ij = self.beta1 * m_ij + (1 - self.beta1) * grad_ij
					self.m[paramName][i][j] = m_ij

					local v_ij = self.v[paramName][i][j]
					v_ij = self.beta2 * v_ij + (1 - self.beta2) * (grad_ij * grad_ij)
					self.v[paramName][i][j] = v_ij

					local mHat = m_ij / (1 - self.beta1 ^ self.t)
					local vHat = v_ij / (1 - self.beta2 ^ self.t)

					W[i][j] = W[i][j] - lr * mHat / (math.sqrt(vHat) + self.epsilon)
					count += 1
				end
			end
		else
			if not self.m[paramName] then
				self.m[paramName] = {}
				self.v[paramName] = {}
				for i, _ in ipairs(W) do
					self.m[paramName][i] = 0
					self.v[paramName][i] = 0
				end
			end

			for i, _ in ipairs(W) do
				local grad_ij = G[i] or 0

				local m_ij = self.m[paramName][i]
				m_ij = self.beta1 * m_ij + (1 - self.beta1) * grad_ij
				self.m[paramName][i] = m_ij

				local v_ij = self.v[paramName][i]
				v_ij = self.beta2 * v_ij + (1 - self.beta2) * (grad_ij * grad_ij)
				self.v[paramName][i] = v_ij

				local mHat = m_ij / (1 - self.beta1 ^ self.t)
				local vHat = v_ij / (1 - self.beta2 ^ self.t)

				W[i] = W[i] - lr * mHat / (math.sqrt(vHat) + self.epsilon)
				count += 1
			end
		end
	end

	return count
end

-- Global gradient norm clipping.  Computes the L2 norm across ALL parameter gradients,
-- scales every gradient down if the norm exceeds maxNorm.  Returns the pre-clip norm.
function Optimizer:clipGradNorm(
	parameters: { [string]: { value: any, grad: any } },
	maxNorm: number
): number
	local totalSq = 0
	for _, paramStruct in pairs(parameters) do
		local G = paramStruct.grad
		if type(G[1]) == "table" then
			for _, row in ipairs(G) do
				for _, g in ipairs(row) do
					totalSq += g * g
				end
			end
		else
			for _, g in ipairs(G) do
				totalSq += g * g
			end
		end
	end

	local norm = math.sqrt(totalSq)
	if norm > maxNorm then
		local scale = maxNorm / norm
		for _, paramStruct in pairs(parameters) do
			local G = paramStruct.grad
			if type(G[1]) == "table" then
				for _, row in ipairs(G) do
					for j, g in ipairs(row) do
						row[j] = g * scale
					end
				end
			else
				for i, g in ipairs(G) do
					G[i] = g * scale
				end
			end
		end
	end

	return norm
end

-- Learning-rate schedule: linear warmup for warmupEpochs, then cosine decay to baseLR/10.
function Optimizer:getLR(epoch: number, totalEpochs: number, baseLR: number, warmupEpochs: number?): number
	local warmup = warmupEpochs or math.max(1, math.floor(totalEpochs * 0.05))
	if epoch <= warmup then
		return baseLR * (epoch / warmup)
	end
	local progress = (epoch - warmup) / math.max(1, totalEpochs - warmup)
	local cosineDecay = 0.5 * (1 + math.cos(math.pi * progress))
	return baseLR * 0.1 + baseLR * 0.9 * cosineDecay
end

return Optimizer
