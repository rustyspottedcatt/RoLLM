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
	forwardCached: (self: MultiHeadAttentionClass, x: Matrix, mask: Matrix?) -> Matrix,
	clearKVCache: (self: MultiHeadAttentionClass) -> (),
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

	self.Wq = LinearAlgebra:xavierMatrix(dModel, dModel)
	self.Wk = LinearAlgebra:xavierMatrix(dModel, dModel)
	self.Wv = LinearAlgebra:xavierMatrix(dModel, dModel)
	self.Wo = LinearAlgebra:xavierMatrix(dModel, dModel)

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

			-- Use cached attn (softmax output) to avoid recomputing exp()
			local weightedSum = 0
			for k = 1, seqLen do
				weightedSum += attn[i][k] * dAttn[i][k]
			end

			for j = 1, seqLen do
				dscores[i][j] = attn[i][j] * (dAttn[i][j] - weightedSum)
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

	local xT = LinearAlgebra:transpose(self.cache.x)
	self.gradWq = LinearAlgebra:mm(xT, gradQ_full)
	self.gradWk = LinearAlgebra:mm(xT, gradK_full)
	self.gradWv = LinearAlgebra:mm(xT, gradV_full)

	-- 7) dL/dx: project merged head‐gradients back through Wq, Wk, Wv
	local gradX_from_Q = LinearAlgebra:mm(gradQ_full, LinearAlgebra:transpose(self.Wq))
	local gradX_from_K = LinearAlgebra:mm(gradK_full, LinearAlgebra:transpose(self.Wk))
	local gradX_from_V = LinearAlgebra:mm(gradV_full, LinearAlgebra:transpose(self.Wv))

	local gradX = LinearAlgebra:add(LinearAlgebra:add(gradX_from_Q, gradX_from_K), gradX_from_V)

	return gradX
end

-- Clear the KV cache (call before starting a new generation sequence).
function MultiHeadAttention:clearKVCache()
	self.kvCacheK = nil
	self.kvCacheV = nil
end

-- Inference-only forward with incremental KV cache.
-- On the first call (cache empty), x may be the full prompt and mask should be the
-- causal mask.  On subsequent single-token calls, mask should be nil.
-- Returns output of shape (#x × dModel).
function MultiHeadAttention:forwardCached(x: Matrix, mask: Matrix?): Matrix
	local qLen  = #x
	local scale = math.sqrt(self.dHead)

	local Q     = LinearAlgebra:mm(x, self.Wq)   -- qLen × dModel
	local K_new = LinearAlgebra:mm(x, self.Wk)   -- qLen × dModel
	local V_new = LinearAlgebra:mm(x, self.Wv)   -- qLen × dModel

	-- Extend KV cache along sequence dimension
	if not self.kvCacheK then
		self.kvCacheK = K_new
		self.kvCacheV = V_new
	else
		local K_full: Matrix = {}
		local V_full: Matrix = {}
		for _, row in ipairs(self.kvCacheK) do table.insert(K_full, row) end
		for _, row in ipairs(K_new)         do table.insert(K_full, row) end
		for _, row in ipairs(self.kvCacheV) do table.insert(V_full, row) end
		for _, row in ipairs(V_new)         do table.insert(V_full, row) end
		self.kvCacheK = K_full
		self.kvCacheV = V_full
	end

	local K_full = self.kvCacheK
	local V_full = self.kvCacheV
	local kvLen  = #K_full

	local headsOut: { Matrix } = {}
	for h = 1, self.numHeads do
		local dOff = (h - 1) * self.dHead

		local Q_h: Matrix = {}
		for i = 1, qLen do
			Q_h[i] = {}
			for d = 1, self.dHead do Q_h[i][d] = Q[i][dOff + d] end
		end

		local K_h: Matrix = {}
		local V_h: Matrix = {}
		for i = 1, kvLen do
			K_h[i] = {}
			V_h[i] = {}
			for d = 1, self.dHead do
				K_h[i][d] = K_full[i][dOff + d]
				V_h[i][d] = V_full[i][dOff + d]
			end
		end

		-- Cross-attention scores: (qLen × kvLen)
		local Kt     = LinearAlgebra:transpose(K_h)
		local scores = LinearAlgebra:mm(Q_h, Kt)

		for i = 1, qLen do
			for j = 1, kvLen do
				scores[i][j] = scores[i][j] / scale
			end
		end

		-- Apply causal mask if provided (initial prompt fill)
		if mask then
			for i = 1, qLen do
				for j = 1, math.min(kvLen, qLen) do
					if mask[i] and mask[i][j] and mask[i][j] < 0 then
						scores[i][j] = -1e9
					end
				end
			end
		end

		-- Row-wise softmax over key dimension
		for i = 1, qLen do
			local rowMax = -math.huge
			for j = 1, kvLen do
				if scores[i][j] > rowMax then rowMax = scores[i][j] end
			end
			local sumExp = 0
			for j = 1, kvLen do
				scores[i][j] = math.exp(scores[i][j] - rowMax)
				sumExp += scores[i][j]
			end
			for j = 1, kvLen do
				scores[i][j] = scores[i][j] / sumExp
			end
		end

		-- attn (qLen × kvLen) @ V_h (kvLen × dHead) → (qLen × dHead)
		headsOut[h] = LinearAlgebra:mm(scores, V_h)
	end

	-- Merge heads into (qLen × dModel)
	local merged: Matrix = {}
	for i = 1, qLen do
		merged[i] = {}
		for h = 1, self.numHeads do
			local dOff = (h - 1) * self.dHead
			for d = 1, self.dHead do
				merged[i][dOff + d] = headsOut[h][i][d]
			end
		end
	end

	return LinearAlgebra:mm(merged, self.Wo)
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
