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
	m: { [string]: { [number]: { [number]: number } } },
	v: { [string]: { [number]: { [number]: number } } },
	t: number,
	updateParameters: (self: OptimizerClass, parameters: { any }, learningRate: number) -> any,
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

function Optimizer:updateParameters(
	parameters: { [string]: { value: { { number } }, grad: { { number } } } },
	learningRate: number?
): nil
	self.t = self.t + 1
	local lr = learningRate or self.learningRate

	for paramName, paramStruct in pairs(parameters) do
		local W = paramStruct.value
		local G = paramStruct.grad

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
		
		-- adam
		for i, row in ipairs(W) do
			for j, _ in ipairs(row) do
				local grad_ij = G[i][j] or 0

				local m_ij = self.m[paramName][i][j]
				m_ij = self.beta1 * m_ij + (1 - self.beta1) * grad_ij
				self.m[paramName][i][j] = m_ij

				local v_ij = self.v[paramName][i][j]
				v_ij = self.beta2 * v_ij + (1 - self.beta2) * (grad_ij * grad_ij)
				self.v[paramName][i][j] = v_ij

				local mHat = m_ij / (1 - self.beta1 ^ self.t)
				local vHat = v_ij / (1 - self.beta2 ^ self.t)

				W[i][j] = W[i][j] - lr * mHat / (math.sqrt(vHat) + self.epsilon)
			end
		end
	end

	return nil
end

return Optimizer
