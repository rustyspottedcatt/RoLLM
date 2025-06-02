--[[
    MultiHeadAttention Module

    @module MultiHeadAttention
    @type MultiHeadAttentionClass
]]

local Types = require(script.Parent.Parent.lib.types)
local LinearAlgebraModule = require(script.Parent.LinearAlgebra)

export type Matrix = Types.Matrix

export type MultiHeadAttentionClass = {
	new: (dModel: number, numHeads: number) -> MultiHeadAttentionClass,
	dModel: number,
	numHeads: number,
	dHead: number,
	Wq: Matrix,
	Wk: Matrix,
	Wv: Matrix,
	Wo: Matrix,
	forward: (self: MultiHeadAttentionClass, x: Matrix, mask: Matrix?) -> Matrix,
	backward: (self: MultiHeadAttentionClass, gradOutput: Matrix) -> Matrix,
	getParameters: (self: MultiHeadAttentionClass) -> { Wq: Matrix, Wk: Matrix, Wv: Matrix, Wo: Matrix },
	setParameters: (self: MultiHeadAttentionClass, params: { Wq: Matrix, Wk: Matrix, Wv: Matrix, Wo: Matrix }) -> nil,
}

local MultiHeadAttention = {}
MultiHeadAttention.__index = MultiHeadAttention

local LinearAlgebra = LinearAlgebraModule.new()

local function splitHeads(X: Matrix, numHeads: number, dHead: number): { Matrix }
	local seqLen = #X
	local heads: { Matrix } = {}
	for h = 1, numHeads do
		heads[h] = {}
		for i = 1, seqLen do
			local row: { number } = {}
			for d = 1, dHead do
				local col = (h - 1) * dHead + d
				row[d] = X[i][col]
			end
			heads[h][i] = row
		end
	end
	return heads
end

local function mergeHeads(heads: { Matrix }, numHeads: number, dHead: number): Matrix
	local seqLen = #heads[1]
	local merged: Matrix = {}
	for i = 1, seqLen do
		merged[i] = {}
		for h = 1, numHeads do
			for d = 1, dHead do
				table.insert(merged[i], heads[h][i][d])
			end
		end
	end
	return merged
end

local function scaledDotProductAttention(Q: Matrix, K: Matrix, V: Matrix, mask: Matrix?): { Matrix & Matrix }
	local seqLen = #Q
	local dHead = #Q[1]

	local Kt = LinearAlgebra:transpose(K)
	local scores = LinearAlgebra:mm(Q, Kt)

	local scale = math.sqrt(dHead)
	for i = 1, seqLen do
		for j = 1, seqLen do
			scores[i][j] = scores[i][j] / scale
		end
	end

	if mask then
		for i = 1, seqLen do
			for j = 1, seqLen do
				if mask[i][j] < 0 then
					scores[i][j] = -1e9
				end
			end
		end
	end

	local attn = LinearAlgebra:softmax(scores)
	local out = LinearAlgebra:mm(attn, V)

	-- Cache for backward pass
	local cache = {
		Q = Q,
		K = K,
		V = V,
		scores = scores,
		attn = attn,
		mask = mask,
	}

	return out, cache
end

function MultiHeadAttention.new(dModel: number, numHeads: number): MultiHeadAttentionClass
	local self = setmetatable({}, MultiHeadAttention)
	self.dModel = dModel
	self.numHeads = numHeads

	assert(dModel % numHeads == 0, "dModel must be divisible by numHeads")
	self.dHead = dModel / numHeads

	self.Wq = LinearAlgebra:randomMatrix(dModel, dModel, 0.01)
	self.Wk = LinearAlgebra:randomMatrix(dModel, dModel, 0.01)
	self.Wv = LinearAlgebra:randomMatrix(dModel, dModel, 0.01)
	self.Wo = LinearAlgebra:randomMatrix(dModel, dModel, 0.01)

	self.gradWq = LinearAlgebra:zeros(dModel, dModel)
	self.gradWk = LinearAlgebra:zeros(dModel, dModel)
	self.gradWv = LinearAlgebra:zeros(dModel, dModel)
	self.gradWo = LinearAlgebra:zeros(dModel, dModel)

	self.cache = {}

	return self
end

function MultiHeadAttention:forward(x: Matrix, mask: Matrix?): Matrix
	local Q = LinearAlgebra:mm(x, self.Wq)
	local K = LinearAlgebra:mm(x, self.Wk)
	local V = LinearAlgebra:mm(x, self.Wv)

	local Q_heads = splitHeads(Q, self.numHeads, self.dHead)
	local K_heads = splitHeads(K, self.numHeads, self.dHead)
	local V_heads = splitHeads(V, self.numHeads, self.dHead)

	local headsOut: { Matrix } = {}
	local attentions: { any } = {}

	for h = 1, self.numHeads do
		local headOut, headCache = scaledDotProductAttention(Q_heads[h], K_heads[h], V_heads[h], mask)
		headsOut[h] = headOut
		attentions[h] = headCache
	end

	local merged = mergeHeads(headsOut, self.numHeads, self.dHead)
	local out = LinearAlgebra:mm(merged, self.Wo)

	self.cache = {
		x = x,
		Q = Q,
		K = K,
		V = V,
		Q_heads = Q_heads,
		K_heads = K_heads,
		V_heads = V_heads,
		headsOut = headsOut,
		attentions = attentions,
		mask = mask,
		merged = merged,
		out = out,
	}

	return out
end

function MultiHeadAttention:backward(gradOutput: Matrix): Matrix
	local gradWo = LinearAlgebra:mm(LinearAlgebra:transpose(self.cache.merged), gradOutput)
	self.gradWo = gradWo

	local gradMerged = LinearAlgebra:mm(gradOutput, LinearAlgebra:transpose(self.Wo))

	local gradHeadsOut: { Matrix } = {}
	for h = 1, self.numHeads do
		gradHeadsOut[h] = {}
		for i = 1, #gradMerged do
			gradHeadsOut[h][i] = {}
			for d = 1, self.dHead do
				gradHeadsOut[h][i][d] = gradMerged[i][(h - 1) * self.dHead + d]
			end
		end
	end

	local gradQ_heads: { Matrix } = {}
	local gradK_heads: { Matrix } = {}
	local gradV_heads: { Matrix } = {}

	for h = 1, self.numHeads do
		local headCache = self.cache.attentions[h]
		local Q_h = headCache.Q
		local K_h = headCache.K
		local V_h = headCache.V
		local scores = headCache.scores
		local attn = headCache.attn
		local mask_h = headCache.mask

		local gradOut = gradHeadsOut[h]

		-- gradV_h = attnᵀ ⋅ gradOut
		local gradV_h = LinearAlgebra:mm(LinearAlgebra:transpose(attn), gradOut)

		-- dAttn = gradOut ⋅ V_hᵀ
		local dAttn = LinearAlgebra:mm(gradOut, LinearAlgebra:transpose(V_h))

		-- Compute dscores = softmax_gradient(scores, dAttn), then zero out masked positions
		local seqLen = #scores
		local dscores = {}
		for i = 1, seqLen do
			dscores[i] = {}

			-- 1) Compute sumExp = ∑ₖ exp(scores[i][k])
			local sumExp = 0
			for k = 1, seqLen do
				sumExp += math.exp(scores[i][k])
			end

			-- 2) Precompute “weighted sum” = ∑ₖ softmax[i][k] * dAttn[i][k]
			local weightedSum = 0
			for k = 1, seqLen do
				local sk = math.exp(scores[i][k]) / sumExp
				weightedSum += sk * dAttn[i][k]
			end

			-- 3) Now compute ∂L/∂scores[i][j] = softmax[i][j] * (dAttn[i][j] – weightedSum)
			for j = 1, seqLen do
				local sj = math.exp(scores[i][j]) / sumExp
				dscores[i][j] = sj * (dAttn[i][j] - weightedSum)
				-- 4) Zero out masked positions
				if mask_h and mask_h[i][j] < 0 then
					dscores[i][j] = 0
				end
			end
		end

		-- gradQ_h = dscores ⋅ K_h, then divide by √dHead
		local scale = math.sqrt(self.dHead)
		local scaledDscores = {}
		for i = 1, seqLen do
			scaledDscores[i] = {}
			for j = 1, seqLen do -- scores is (seqLen × seqLen), so loop j=1..seqLen
				scaledDscores[i][j] = dscores[i][j] / scale
			end
		end
		local gradQ_h = LinearAlgebra:mm(scaledDscores, K_h)

		-- gradK_h = dscoresᵀ ⋅ Q_h, then divide by √dHead
		local dscoresT = LinearAlgebra:transpose(scaledDscores)
		local gradK_h = LinearAlgebra:mm(dscoresT, Q_h)

		gradQ_heads[h] = gradQ_h
		gradK_heads[h] = gradK_h
		gradV_heads[h] = gradV_h
	end

	local seqLen = #self.cache.x

	local gradQ_full = LinearAlgebra:zeros(seqLen, self.dModel)
	for h = 1, self.numHeads do
		for i = 1, seqLen do
			for d = 1, self.dHead do
				gradQ_full[i][(h - 1) * self.dHead + d] = gradQ_heads[h][i][d]
			end
		end
	end

	-- 5b) Same for gradK_full
	local gradK_full = LinearAlgebra:zeros(seqLen, self.dModel)
	for h = 1, self.numHeads do
		for i = 1, seqLen do
			for d = 1, self.dHead do
				gradK_full[i][(h - 1) * self.dHead + d] = gradK_heads[h][i][d]
			end
		end
	end

	-- 5c) Same for gradV_full
	local gradV_full = LinearAlgebra:zeros(seqLen, self.dModel)
	for h = 1, self.numHeads do
		for i = 1, seqLen do
			for d = 1, self.dHead do
				gradV_full[i][(h - 1) * self.dHead + d] = gradV_heads[h][i][d]
			end
		end
	end

	for h = 1, self.numHeads do
		self.gradWq =
			LinearAlgebra:add(self.gradWq, LinearAlgebra:mm(LinearAlgebra:transpose(self.cache.x), gradQ_heads[h]))
		self.gradWk =
			LinearAlgebra:add(self.gradWk, LinearAlgebra:mm(LinearAlgebra:transpose(self.cache.x), gradK_heads[h]))
		self.gradWv =
			LinearAlgebra:add(self.gradWv, LinearAlgebra:mm(LinearAlgebra:transpose(self.cache.x), gradV_heads[h]))
	end

	-- 7) dL/dx: project merged head‐gradients back through Wq, Wk, Wv
	local gradX_from_Q = LinearAlgebra:mm(gradQ_full, LinearAlgebra:transpose(self.Wq))
	local gradX_from_K = LinearAlgebra:mm(gradK_full, LinearAlgebra:transpose(self.Wk))
	local gradX_from_V = LinearAlgebra:mm(gradV_full, LinearAlgebra:transpose(self.Wv))

	local gradX = LinearAlgebra:add(LinearAlgebra:add(gradX_from_Q, gradX_from_K), gradX_from_V)

	return gradX
end

function MultiHeadAttention:getParameters(): { Wq: Matrix, Wk: Matrix, Wv: Matrix, Wo: Matrix }
	return {
		Wq = self.Wq,
		Wk = self.Wk,
		Wv = self.Wv,
		Wo = self.Wo,
	}
end

function MultiHeadAttention:setParameters(params: { Wq: Matrix, Wk: Matrix, Wv: Matrix, Wo: Matrix })
	self.Wq = params.Wq
	self.Wk = params.Wk
	self.Wv = params.Wv
	self.Wo = params.Wo
end

return MultiHeadAttention
