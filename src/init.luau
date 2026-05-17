local ReplicatedStorage = game:GetService("ReplicatedStorage")
--[[
    https://www.roblox.com/users/1539582829/profile
    https://twitter.com/zzen_a

    MIT License

    Copyright (c) 2024 rustyspotted

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
]]

local Types = require(script.lib.types)
local Optimizer = require(script.lib.Optimizer)
local CrossEntropyLoss = require(script.lib.CrossEntropyLoss)

local LinearAlgebra = require(script.components.LinearAlgebra)
local Tokenizer = require(script.components.Tokenizer)
local Embedding = require(script.components.Embedding)
local MultiHeadAttention = require(script.components.MultiHeadAttention)
local FeedForward = require(script.components.FeedForward)
local LayerNorm = require(script.components.LayerNorm)
local TransformerBlock = require(script.components.TransformerBlock)
local TransformerModel = require(script.components.TransformerModel)
local CharTokenizer = require(script.components.CharTokenizer)
local BPETokenizer = require(script.components.BPETokenizer)

local RoLLM = {}

export type TrainingEpoch = Types.TrainingEpoch

--[=[
    @prop Presets Folder
    @within RoLLM
    @readonly
    References the Presets folder.
]=]
RoLLM.Presets = ReplicatedStorage:WaitForChild("Packages") :: Folder

RoLLM.Promise = require(RoLLM.Presets.promise)
RoLLM.Signal = require(RoLLM.Presets.signal)
RoLLM.Types = Types
RoLLM.Optimizer = Optimizer
RoLLM.CrossEntropyLoss = CrossEntropyLoss
RoLLM.LinearAlgebra = LinearAlgebra
RoLLM.Tokenizer = Tokenizer
RoLLM.Embedding = Embedding
RoLLM.MultiHeadAttention = MultiHeadAttention
RoLLM.FeedForward = FeedForward
RoLLM.LayerNorm = LayerNorm
RoLLM.TransformerBlock = TransformerBlock
RoLLM.TransformerModel = TransformerModel
RoLLM.CharTokenizer = CharTokenizer
RoLLM.BPETokenizer = BPETokenizer

--[=[
    @function chunkString
    @within RoLLM
    @param bigString string - The string to be chunked.
    @param chunkSize number - The size of each chunk.
    @return {string} - A table containing string chunks.
]=]
local function chunkString(bigString: string, chunkSize: number): { string }
	local chunks: { string } = {}
	local idx = 1
	while idx <= #bigString do
		local chunk = string.sub(bigString, idx, idx + chunkSize - 1)
		table.insert(chunks, chunk)
		idx += chunkSize
	end
	return chunks
end

local function validateConfig(config: Types.TransformerConfig)
	assert(config.dModel ~= nil and config.dModel > 0, "TransformerConfig.dModel must be greater than 0")
	assert(config.numHeads ~= nil and config.numHeads > 0, "TransformerConfig.numHeads must be greater than 0")
	assert(config.dFF ~= nil and config.dFF > 0, "TransformerConfig.dFF must be greater than 0")
	assert(config.numLayers ~= nil and config.numLayers > 0, "TransformerConfig.numLayers must be greater than 0")
	assert(config.maxSeqLen == nil or config.maxSeqLen > 1, "TransformerConfig.maxSeqLen must be greater than 1")
	assert(config.dModel % config.numHeads == 0, "TransformerConfig.dModel must be divisible by numHeads")
end

local function forEachTextChunk(textData: string | { string }, chunkSize: number, callback: (string) -> ())
	if type(textData) == "table" then
		assert(#textData > 0, "textData must contain at least one string")
		for _, singleString in ipairs(textData) do
			assert(type(singleString) == "string" and #singleString > 0, "textData entries must be non-empty strings")
			if #singleString > chunkSize then
				for _, chunk in ipairs(chunkString(singleString, chunkSize)) do
					callback(chunk)
				end
			else
				callback(singleString)
			end
		end
	else
		assert(type(textData) == "string" and #textData > 0, "textData must be a non-empty string or table of strings")
		if #textData > chunkSize then
			for _, chunk in ipairs(chunkString(textData, chunkSize)) do
				callback(chunk)
			end
		else
			callback(textData)
		end
	end
end

local function trimTokensToContext(tokens: { number }, maxSeqLen: number): { number }
	assert(#tokens > 0, "Input text must produce at least one token")
	if #tokens <= maxSeqLen then
		return tokens
	end

	local trimmed: { number } = {}
	local startIndex = #tokens - maxSeqLen + 1
	for i = startIndex, #tokens do
		table.insert(trimmed, tokens[i])
	end
	return trimmed
end

--[=[
    Creates a "ready-to-use" LLM instance with a tokenizer and model.

    @function new
    @within RoLLM
    @param textData string | {string} - Text data for building the vocabulary.
    @param config Types.TransformerConfig - Configuration table for the Transformer.
    @param chunkSize number? - Optional chunk size for processing text.
    @return {any} - An LLM object with prediction, generation, and training methods.
]=]
function RoLLM.new(textData: string | { string }, config: Types.TransformerConfig, chunkSize: number?): { any }
	if config == nil then
		error("TransformerConfig is required when constructing RoLLM")
	end
	if type(config) ~= "table" then
		error("TransformerConfig must be a table")
	end
	validateConfig(config)

	chunkSize = chunkSize or 500000 :: number
	assert(chunkSize > 0, "chunkSize must be greater than 0")
	local tokenizerMode: string = config.tokenizerMode or "char"
	assert(tokenizerMode == "char" or tokenizerMode == "bpe", "tokenizerMode must be 'char' or 'bpe'")

	local tokenizer
	if tokenizerMode == "char" then
		tokenizer = CharTokenizer.new()
	else
		local bpeTok: BPETokenizer.BPETokenizerClass = BPETokenizer.new()
		if config.externalVocabURL then
			bpeTok:loadExternalVocab(config.externalVocabURL)
		end
		tokenizer = bpeTok
	end

	local function addTextForChar(str: string)
		if tokenizerMode == "char" then
			tokenizer:buildVocabFromText(str)
		else
			tokenizer:textToTokens(str)
		end
	end

	forEachTextChunk(textData, chunkSize :: number, addTextForChar)

	local vocabSize: number = tokenizer:getVocabSize()
	if vocabSize == 0 then
		error("Vocab size is 0. Did you load data or external vocab properly?")
	end
	config.vocabSize = vocabSize

	config.maxSeqLen = config.maxSeqLen or 128
	local maxSeqLen = config.maxSeqLen
	local model: TransformerModel.TransformerModelClass = TransformerModel.new(config)

	local self = {}

	local function textToTokens(str: string): { number }
		return tokenizer:textToTokens(str)
	end

	local function tokensToText(toks: { number }): string
		return tokenizer:tokensToText(toks)
	end

	function self:predict(inputStr: string): string
		local tokens = trimTokensToContext(textToTokens(inputStr), maxSeqLen)
		local nextID = model:predictNextToken(tokens)
		return tokensToText({ nextID })
	end

	function self:generate(inputStr: string, numTokens: number): string
		assert(numTokens >= 0, "numTokens must be greater than or equal to 0")
		local tokens = textToTokens(inputStr)
		local outputTokens = { table.unpack(tokens) }
		for _ = 1, numTokens do
			local context = trimTokensToContext(tokens, maxSeqLen)
			local nxt = model:predictNextToken(context)
			table.insert(tokens, nxt)
			table.insert(outputTokens, nxt)
		end
		return tokensToText(outputTokens)
	end

	function self:predictTemperature(inputStr: string, temperature: number): string
		assert(temperature >= 0, "temperature must be greater than or equal to 0")
		local tokens = trimTokensToContext(textToTokens(inputStr), maxSeqLen)
		local nextID = model:predictNextTokenTemperature(tokens, temperature)
		return tokensToText({ nextID })
	end

	function self:generateTemperature(inputStr: string, numTokens: number, temperature: number): string
		assert(numTokens >= 0, "numTokens must be greater than or equal to 0")
		assert(temperature >= 0, "temperature must be greater than or equal to 0")
		local tokens = textToTokens(inputStr)
		local outputTokens = { table.unpack(tokens) }
		for _ = 1, numTokens do
			local context = trimTokensToContext(tokens, maxSeqLen)
			local nxt = model:predictNextTokenTemperature(context, temperature)
			table.insert(tokens, nxt)
			table.insert(outputTokens, nxt)
		end
		return tokensToText(outputTokens)
	end

	--[=[
        Trains the Transformer model using provided training data.

        @function trainModel
        @within RoLLM
        @param trainingData {string} - A table of training strings.
        @param epochs number - Number of training epochs.
        @param learningRate number - Learning rate for the optimizer.
        @param options Types.TrainingOptions? - Optional training options (e.g. yieldEverySamples).
        @return {TrainingEpoch}
    ]=]
	function self:trainModel(trainingData: { string }, epochs: number, learningRate: number, options: Types.TrainingOptions?): { TrainingEpoch }
		if type(trainingData) ~= "table" then
			error("trainingData must be a table of strings")
		end
		assert(#trainingData > 0, "trainingData must contain at least one string")
		assert(epochs > 0, "epochs must be greater than 0")
		assert(learningRate > 0, "learningRate must be greater than 0")

		local optimizerInstance = Optimizer.new(learningRate)
		local lossFunction = CrossEntropyLoss.new()
		local history: { TrainingEpoch } = {}
		local tokenizedTrainingData: { { number } } = {}
		local yieldEvery: number? = options and options.yieldEverySamples
		local sampleCount = 0
		local startTime = os.clock()

		for _, data in ipairs(trainingData) do
			local tokens = trimTokensToContext(textToTokens(data), maxSeqLen)
			if #tokens < 2 then
				error(string.format("Input sequence is too short for training. Tokens: %s", table.concat(tokens, ", ")))
			end
			table.insert(tokenizedTrainingData, tokens)
		end

		for epoch = 1, epochs do
			local totalLoss = 0
			local samples = 0

			for _, tokens in ipairs(tokenizedTrainingData) do
				local logitsFull = model:forward(tokens)
				local seqLen = #logitsFull

				local truncatedLogits: { { number } } = {}
				for i = 1, seqLen - 1 do
					truncatedLogits[i] = logitsFull[i]
				end

				local targets: { number } = {}
				for i = 2, #tokens do
					table.insert(targets, tokens[i])
				end

				local loss = lossFunction:compute(truncatedLogits, targets)
				totalLoss += loss
				samples += 1

				local gradTruncated: { { number } } = lossFunction:backward(truncatedLogits, targets)

				local paddedGrad: { { number } } = {}
				for i = 1, seqLen - 1 do
					paddedGrad[i] = gradTruncated[i]
				end
				paddedGrad[seqLen] = {}
				for j = 1, vocabSize do
					paddedGrad[seqLen][j] = 0
				end

				local backResults = model:backward(paddedGrad)
				local gradFinalProj = backResults.gradFinalProj
				local gradEmbeddingWeights = backResults.gradEmbeddingWeights

				local allParams: { [string]: { value: any, grad: any } } = {}

				allParams["finalProj"] = {
					value = model.finalProj,
					grad = gradFinalProj,
				}

				allParams["embeddingWeights"] = {
					value = model.embedding.weights,
					grad = gradEmbeddingWeights,
				}

				for idx, block in ipairs(model.blocks) do
					local blockParams = block:getParameters()
					local blockPrefix = string.format("block%d", idx)

					allParams[blockPrefix .. "_mha_Wq"] = {
						value = blockParams.mha.Wq,
						grad = blockParams.mha.gradWq or {},
					}
					allParams[blockPrefix .. "_mha_Wk"] = {
						value = blockParams.mha.Wk,
						grad = blockParams.mha.gradWk or {},
					}
					allParams[blockPrefix .. "_mha_Wv"] = {
						value = blockParams.mha.Wv,
						grad = blockParams.mha.gradWv or {},
					}
					allParams[blockPrefix .. "_mha_Wo"] = {
						value = blockParams.mha.Wo,
						grad = blockParams.mha.gradWo or {},
					}

					allParams[blockPrefix .. "_ff_W1"] = {
						value = blockParams.ff.W1,
						grad = blockParams.ff.gradW1 or {},
					}
					allParams[blockPrefix .. "_ff_b1"] = {
						value = blockParams.ff.b1,
						grad = blockParams.ff.gradB1 or {},
					}
					allParams[blockPrefix .. "_ff_W2"] = {
						value = blockParams.ff.W2,
						grad = blockParams.ff.gradW2 or {},
					}
					allParams[blockPrefix .. "_ff_b2"] = {
						value = blockParams.ff.b2,
						grad = blockParams.ff.gradB2 or {},
					}

					allParams[blockPrefix .. "_norm1_gamma"] = {
						value = blockParams.norm1.gamma,
						grad = blockParams.norm1.gradGamma or {},
					}
					allParams[blockPrefix .. "_norm1_beta"] = {
						value = blockParams.norm1.beta,
						grad = blockParams.norm1.gradBeta or {},
					}
					allParams[blockPrefix .. "_norm2_gamma"] = {
						value = blockParams.norm2.gamma,
						grad = blockParams.norm2.gradGamma or {},
					}
					allParams[blockPrefix .. "_norm2_beta"] = {
						value = blockParams.norm2.beta,
						grad = blockParams.norm2.gradBeta or {},
					}
				end

				optimizerInstance:updateParameters(allParams)
				sampleCount += 1
				if yieldEvery and sampleCount % yieldEvery == 0 then
					task.wait()
				end
			end

			local averageLoss = totalLoss / math.max(samples, 1)
			table.insert(history, {
				epoch = epoch,
				loss = totalLoss,
				averageLoss = averageLoss,
				samples = samples,
			})
			if options and options.onEpochComplete then
				options.onEpochComplete(epoch, epochs, averageLoss, os.clock() - startTime)
			end
		end

		return history
	end

	function self:getParameterCount(): number
		local count = 0
		count += #model.embedding.weights * #model.embedding.weights[1]
		count += #model.finalProj * #model.finalProj[1]
		for _, block in ipairs(model.blocks) do
			local params = block:getParameters()
			count += #params.mha.Wq * #params.mha.Wq[1]
			count += #params.mha.Wk * #params.mha.Wk[1]
			count += #params.mha.Wv * #params.mha.Wv[1]
			count += #params.mha.Wo * #params.mha.Wo[1]
			count += #params.ff.W1 * #params.ff.W1[1]
			count += #params.ff.b1
			count += #params.ff.W2 * #params.ff.W2[1]
			count += #params.ff.b2
			count += #params.norm1.gamma + #params.norm1.beta
			count += #params.norm2.gamma + #params.norm2.beta
		end
		return count
	end

	function self:saveModel(storeKey: string): boolean
		local DataStoreService = game:GetService("DataStoreService")
		local store = DataStoreService:GetDataStore("RoLLM")

		local serialized = {
			version = 1,
			embeddingWeights = model.embedding.weights,
			finalProj = model.finalProj,
			blocks = {},
		}

		for i, block in ipairs(model.blocks) do
			local bp = block:getParameters()
			serialized.blocks[i] = {
				mha = { Wq = bp.mha.Wq, Wk = bp.mha.Wk, Wv = bp.mha.Wv, Wo = bp.mha.Wo },
				ff = { W1 = bp.ff.W1, b1 = bp.ff.b1, W2 = bp.ff.W2, b2 = bp.ff.b2 },
				norm1 = { gamma = bp.norm1.gamma, beta = bp.norm1.beta },
				norm2 = { gamma = bp.norm2.gamma, beta = bp.norm2.beta },
			}
		end

		local ok, err = pcall(function()
			store:SetAsync(storeKey, serialized)
		end)

		if not ok then
			warn(string.format("RoLLM.saveModel: DataStore write failed: %s", tostring(err)))
		end

		return ok
	end

	function self:loadModel(storeKey: string): boolean
		local DataStoreService = game:GetService("DataStoreService")
		local store = DataStoreService:GetDataStore("RoLLM")

		local ok, data = pcall(function()
			return store:GetAsync(storeKey)
		end)

		if not ok or data == nil then
			return false
		end

		model.embedding.weights = data.embeddingWeights
		model.finalProj = data.finalProj

		for i, blockData in ipairs(data.blocks) do
			local block = model.blocks[i]
			if block then
				block.mha:setParameters(blockData.mha)
				block.ff:setParameters(blockData.ff)
				block.norm1:setParameters(blockData.norm1)
				block.norm2:setParameters(blockData.norm2)
			end
		end

		return true
	end

	self._tokenizer = tokenizer
	self._model = model

	return self
end

return RoLLM
