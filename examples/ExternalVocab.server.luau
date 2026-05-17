--[[
    ExternalVocab.server.luau
    
    Uses an external BPE vocabulary fetched over HTTP (GPT-2's vocab.json from
    Hugging Face) instead of building a character-level vocab from the corpus.
    
    Why: a char-level model has a vocab of ~50-100 chars. BPE splits text into
    subword tokens, giving a vocab of 50,000+ entries. The model can represent
    more meaning per token, which matters for coherent generation. The tradeoff
    is that the final projection matrix (dModel x vocabSize) becomes enormous —
    only use external vocab with a very small dModel to keep memory reasonable.
    
    Requirements:
      Game Settings > Security > Allow HTTP Requests must be enabled.
      The Hugging Face URL must be reachable from Roblox servers.
      
    Note:
      With a 50K vocab and dModel=32, the embedding alone is 1.6M params.
      Inference works but training is extremely slow. This example only
      demonstrates construction and a single predict call, not full training.
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RoLLM = require(ReplicatedStorage.RoLLM)

local VOCAB_URL = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"

print("Creating model with GPT-2 external vocab...")
print("(This will make an HTTP request to Hugging Face — may take a few seconds)")

local ok, result = pcall(function()
	return RoLLM.new({}, {
		dModel = 16,
		numHeads = 2,
		dFF = 32,
		numLayers = 1,
		maxSeqLen = 16,
		tokenizerMode = "external",
		externalVocabURL = VOCAB_URL,
	})
end)

if not ok then
	warn(string.format([[
ExternalVocab: failed to create model.
Error: %s

Common causes:
  - HTTP Requests are disabled (Game Settings > Security > Allow HTTP Requests)
  - Hugging Face URL is blocked by Roblox's HTTP allowlist
  - Studio is in offline mode
]], tostring(result)))
	return
end

local llm = result

print(string.format("Model created. Vocab size: %d tokens", llm._tokenizer:getVocabSize()))
print(string.format("Parameter count: %d", llm:getParameterCount()))

print("\n--- Single-token prediction (untrained, random weights) ---")
print("Note: with random weights these outputs are meaningless.")
print("They demonstrate that the pipeline runs end-to-end with a large vocab.")

local prompts = { "Hello", "The sky", "Once upon" }
for _, prompt in ipairs(prompts) do
	local ok2, nextTok = pcall(function()
		return llm:predict(prompt)
	end)
	if ok2 then
		print(string.format("  predict(%q) -> token index %d", prompt, nextTok))
	else
		print(string.format("  predict(%q) -> error: %s", prompt, tostring(nextTok)))
	end
end

print([[
  
To actually train with an external vocab you would need either:
  - A very small dModel (8-16) to keep the projection matrix manageable
  - Pre-trained weights loaded via saveModel/loadModel
  - Far more training data than a tiny char-level model needs
]])
