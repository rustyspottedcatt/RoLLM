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

--[=[
    @prop Presets Folder
    @within RoLLM
    @readonly
    References the Presets folder.
]=]
RoLLM.Presets = script.Parent :: Instance

RoLLM.Promise = require(RoLLM.Presets.promise)
RoLLM.Signal = require(RoLLM.Presets.signal)
RoLLM.Types = Types
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

	chunkSize = chunkSize or 500000 :: number
	local tokenizerMode: string = config.tokenizerMode or "char"

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

	if tokenizerMode == "char" then
		if type(textData) == "table" then
			for _, singleString in ipairs(textData) do
				if #singleString > chunkSize then
					local chunks = chunkString(singleString, chunkSize :: number)
					for _, chunk in ipairs(chunks) do
						addTextForChar(chunk)
					end
				else
					addTextForChar(singleString)
				end
			end
		else
			if #textData > chunkSize then
				local chunks = chunkString(textData, chunkSize :: number)
				for _, chunk in ipairs(chunks) do
					addTextForChar(chunk)
				end
			else
				addTextForChar(textData)
			end
		end
	else
		if type(textData) == "table" then
			for _, singleString in ipairs(textData) do
				if #singleString > chunkSize then
					local chunks = chunkString(singleString, chunkSize :: number)
					for _, chunk in ipairs(chunks) do
						addTextForChar(chunk)
					end
				else
					addTextForChar(singleString)
				end
			end
		else
			if #textData > chunkSize then
				local chunks = chunkString(textData, chunkSize :: number)
				for _, chunk in ipairs(chunks) do
					addTextForChar(chunk)
				end
			else
				addTextForChar(textData)
			end
		end
	end

	local vocabSize: number = tokenizer:getVocabSize()
	if vocabSize == 0 then
		error("Vocab size is 0. Did you load data or external vocab properly?")
	end
	config.vocabSize = vocabSize
	print("Final vocab size:", vocabSize)

	config.maxSeqLen = config.maxSeqLen or 128
	local model: TransformerModel.TransformerModelClass = TransformerModel.new(config)

	local self = {}

	local function textToTokens(str: string): { number }
		return tokenizer:textToTokens(str)
	end

	local function tokensToText(toks: { number }): string
		return tokenizer:tokensToText(toks)
	end

	function self:predict(inputStr: string): string
		local tokens = textToTokens(inputStr)
		local nextID = model:predictNextToken(tokens)
		return tokensToText({ nextID })
	end

	function self:generate(inputStr: string, numTokens: number): string
		local tokens = textToTokens(inputStr)
		for _ = 1, numTokens do
			local nxt = model:predictNextToken(tokens)
			table.insert(tokens, nxt)
		end
		return tokensToText(tokens)
	end

	function self:predictTemperature(inputStr: string, temperature: number): string
		local tokens = textToTokens(inputStr)
		local nextID = model:predictNextTokenTemperature(tokens, temperature)
		return tokensToText({ nextID })
	end

	function self:generateTemperature(inputStr: string, numTokens: number, temperature: number): string
		local tokens = textToTokens(inputStr)
		for _ = 1, numTokens do
			local nxt = model:predictNextTokenTemperature(tokens, temperature)
			table.insert(tokens, nxt)
		end
		return tokensToText(tokens)
	end

	--[=[
        Trains the Transformer model using provided training data.

        @function trainModel
        @within RoLLM
        @param trainingData {string} - A table of training strings.
        @param epochs number - Number of training epochs.
        @param learningRate number - Learning rate for the optimizer.
        @return nil
    ]=]
	function self:trainModel(trainingData: { string }, epochs: number, learningRate: number): nil
		if type(trainingData) ~= "table" then
			error("trainingData must be a table of strings")
		end

		local optimizerInstance = Optimizer.new(learningRate)
		local lossFunction = CrossEntropyLoss.new()

		print(string.format("Starting Training: %d epochs, Learning Rate: %f", epochs, learningRate))

		for epoch = 1, epochs do
			local totalLoss = 0

			for _, data in ipairs(trainingData) do
				local tokens = textToTokens(data)
				if #tokens < 2 then
					error(
						string.format(
							"Input sequence is too short for training. Tokens: %s",
							table.concat(tokens, ", ")
						)
					)
				end

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
				local gradEmbeddingOut = backResults.gradEmbedding
				local gradFinalProj = backResults.gradFinalProj
				local gradEmbeddingWeights = backResults.gradEmbeddingWeights

				local allParams: { [string]: { value: { { number } }, grad: { { number } } } } = {}

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

					allParams[string.format("block%d_mha_Wq", idx)] = {
						value = blockParams.mha.Wq,
						grad = blockParams.mha.gradWq or {},
					}
					allParams[string.format("block%d_mha_Wk", idx)] = {
						value = blockParams.mha.Wk,
						grad = blockParams.mha.gradWk or {},
					}
					allParams[string.format("block%d_mha_Wv", idx)] = {
						value = blockParams.mha.Wv,
						grad = blockParams.mha.gradWv or {},
					}
					allParams[string.format("block%d_mha_Wo", idx)] = {
						value = blockParams.mha.Wo,
						grad = blockParams.mha.gradWo or {},
					}

					allParams[string.format("block%d_ff_W1", idx)] = {
						value = blockParams.ff.W1,
						grad = blockParams.ff.gradW1 or {},
					}
					allParams[string.format("block%d_ff_b1", idx)] = {
						value = blockParams.ff.b1,
						grad = blockParams.ff.gradb1 or {},
					}
					allParams[string.format("block%d_ff_W2", idx)] = {
						value = blockParams.ff.W2,
						grad = blockParams.ff.gradW2 or {},
					}
					allParams[string.format("block%d_ff_b2", idx)] = {
						value = blockParams.ff.b2,
						grad = blockParams.ff.gradb2 or {},
					}

					allParams[string.format("block%d_norm1_gamma", idx)] = {
						value = blockParams.norm1.gamma,
						grad = blockParams.norm1.gradGamma or {},
					}
					allParams[string.format("block%d_norm1_beta", idx)] = {
						value = blockParams.norm1.beta,
						grad = blockParams.norm1.gradBeta or {},
					}
					allParams[string.format("block%d_norm2_gamma", idx)] = {
						value = blockParams.norm2.gamma,
						grad = blockParams.norm2.gradGamma or {},
					}
					allParams[string.format("block%d_norm2_beta", idx)] = {
						value = blockParams.norm2.beta,
						grad = blockParams.norm2.gradBeta or {},
					}
				end

				optimizerInstance:updateParameters(allParams)
			end

			print(string.format("Epoch %d/%d, Loss: %.4f", epoch, epochs, totalLoss))
		end

		print("Training Completed.")
		return nil
	end

	self._tokenizer = tokenizer
	self._model = model

	return self
end

return RoLLM
