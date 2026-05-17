--[[
    WorkerAgent.server.luau

    Data-parallel training worker. DO NOT move, rename, or place manually.
    The RoLLM coordinator (init.luau:trainModel) clones this script at runtime
    and parents each clone into a new Actor in ServerScriptService.

    Communication protocol:
      Coordinator -> Worker  :  actor:SendMessage("Work", workerId, resultEvent, modelConfig, snapshot, shard)
      Worker -> Coordinator  :  resultEvent:Fire(workerId, gradAccum, totalLoss, sampleCount)

    snapshot  = flat table { [paramKey]: matrix/vector } — received as a deep copy from Actor messaging
    gradAccum = flat table with the same keys — accumulated over all samples in this shard
]]

local actor = script:FindFirstAncestorOfClass("Actor")
if not actor then
	warn("RoLLM WorkerAgent: not running inside an Actor — exiting.")
	return
end

local RoLLMRoot     = game:GetService("ReplicatedStorage").RoLLM
local TransformerModel = require(RoLLMRoot.components.TransformerModel)
local CrossEntropyLoss = require(RoLLMRoot.lib.CrossEntropyLoss)

local function applySnapshot(mdl, snapshot)
	mdl.finalProj            = snapshot.finalProj
	mdl.embedding.weights    = snapshot.embeddingWeights
	for idx, block in ipairs(mdl.blocks) do
		local p = "block" .. idx
		block.mha.Wq      = snapshot[p .. "_mha_Wq"]
		block.mha.Wk      = snapshot[p .. "_mha_Wk"]
		block.mha.Wv      = snapshot[p .. "_mha_Wv"]
		block.mha.Wo      = snapshot[p .. "_mha_Wo"]
		block.ff.W1       = snapshot[p .. "_ff_W1"]
		block.ff.b1       = snapshot[p .. "_ff_b1"]
		block.ff.W2       = snapshot[p .. "_ff_W2"]
		block.ff.b2       = snapshot[p .. "_ff_b2"]
		block.norm1.gamma = snapshot[p .. "_norm1_gamma"]
		block.norm1.beta  = snapshot[p .. "_norm1_beta"]
		block.norm2.gamma = snapshot[p .. "_norm2_gamma"]
		block.norm2.beta  = snapshot[p .. "_norm2_beta"]
	end
end

actor:BindToMessageParallel("Work", function(
	workerId:    number,
	resultEvent: BindableEvent,
	modelConfig: { [string]: any },
	snapshot:    { [string]: any },
	shard:       { { number } }
)
	local mdl        = TransformerModel.new(modelConfig)
	applySnapshot(mdl, snapshot)

	local lossFunction             = CrossEntropyLoss.new()
	local vocabSize: number        = modelConfig.vocabSize
	local gradAccum: { [string]: any } = {}
	local totalLoss                = 0
	local sampleCount              = 0

	for _, tokens in ipairs(shard) do
		local logitsFull = mdl:forward(tokens)
		local seqLen     = #logitsFull

		local truncatedLogits: { { number } } = {}
		for i = 1, seqLen - 1 do
			truncatedLogits[i] = logitsFull[i]
		end

		local targets: { number } = {}
		for i = 2, #tokens do
			targets[#targets + 1] = tokens[i]
		end

		totalLoss += lossFunction:compute(truncatedLogits, targets)

		local gradTruncated = lossFunction:backward(truncatedLogits, targets)

		local paddedGrad: { { number } } = {}
		for i = 1, seqLen - 1 do paddedGrad[i] = gradTruncated[i] end
		paddedGrad[seqLen] = {}
		for j = 1, vocabSize do paddedGrad[seqLen][j] = 0 end

		local backResults = mdl:backward(paddedGrad)

		local allParams: { [string]: { value: any, grad: any } } = {
			finalProj        = { value = mdl.finalProj,         grad = backResults.gradFinalProj },
			embeddingWeights = { value = mdl.embedding.weights, grad = backResults.gradEmbeddingWeights },
		}
		for idx, block in ipairs(mdl.blocks) do
			local bp = block:getParameters()
			local p  = "block" .. idx
			allParams[p .. "_mha_Wq"]      = { value = bp.mha.Wq,     grad = bp.mha.gradWq     or {} }
			allParams[p .. "_mha_Wk"]      = { value = bp.mha.Wk,     grad = bp.mha.gradWk     or {} }
			allParams[p .. "_mha_Wv"]      = { value = bp.mha.Wv,     grad = bp.mha.gradWv     or {} }
			allParams[p .. "_mha_Wo"]      = { value = bp.mha.Wo,     grad = bp.mha.gradWo     or {} }
			allParams[p .. "_ff_W1"]       = { value = bp.ff.W1,      grad = bp.ff.gradW1      or {} }
			allParams[p .. "_ff_b1"]       = { value = bp.ff.b1,      grad = bp.ff.gradB1      or {} }
			allParams[p .. "_ff_W2"]       = { value = bp.ff.W2,      grad = bp.ff.gradW2      or {} }
			allParams[p .. "_ff_b2"]       = { value = bp.ff.b2,      grad = bp.ff.gradB2      or {} }
			allParams[p .. "_norm1_gamma"] = { value = bp.norm1.gamma, grad = bp.norm1.gradGamma or {} }
			allParams[p .. "_norm1_beta"]  = { value = bp.norm1.beta,  grad = bp.norm1.gradBeta  or {} }
			allParams[p .. "_norm2_gamma"] = { value = bp.norm2.gamma, grad = bp.norm2.gradGamma or {} }
			allParams[p .. "_norm2_beta"]  = { value = bp.norm2.beta,  grad = bp.norm2.gradBeta  or {} }
		end

		for k, paramStruct in pairs(allParams) do
			local g = paramStruct.grad
			if not gradAccum[k] then
				if type(g[1]) == "table" then
					gradAccum[k] = {}
					for i = 1, #g do
						gradAccum[k][i] = {}
						for j = 1, #g[i] do gradAccum[k][i][j] = g[i][j] or 0 end
					end
				else
					gradAccum[k] = {}
					for i = 1, #g do gradAccum[k][i] = g[i] or 0 end
				end
			else
				if type(g[1]) == "table" then
					for i = 1, #g do
						for j = 1, #g[i] do gradAccum[k][i][j] += g[i][j] or 0 end
					end
				else
					for i = 1, #g do gradAccum[k][i] += g[i] or 0 end
				end
			end
		end
		sampleCount += 1
	end

	task.synchronize()
	resultEvent:Fire(workerId, gradAccum, totalLoss, sampleCount)
end)
