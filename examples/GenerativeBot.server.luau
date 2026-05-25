--[[
    GenerativeBot v2 — RoLLM generative-model demo
    ================================================
    Demonstrates all RoLLM v2 features in one place:
      • AdaptiveVocab  — GPT-2 BPE tokens compressed to a 512-token dense vocab
      • CorpusFetcher  — HTTP corpus loading with DataStore cache
      • Xavier init    — via TransformerModel.new (FeedForward + MHA weights)
      • GELU           — via FeedForward (automatic, no config needed)
      • Weight tying   — embedding and output projection share the same matrix
      • Grad clipping  — TrainingOptions.gradClipNorm = 1.0
      • LR schedule    — TrainingOptions.lrSchedule = true (warmup + cosine)
      • KV-cache gen   — self:generateWithKVCache for O(n) inference
      • Top-k / nuc.   — self:generateTopK / self:generateNucleus
      • Multi-key DS   — saveModel / loadModel handle any model size

    How to use in Studio:
      1.  Place this script in ServerScriptService.
      2.  Enable "Allow HTTP Requests" in Game Settings > Security.
      3.  Enable "Studio Access to API Services" in Game Settings > Security.
      4.  Hit Play.  The bot trains once and saves its weights.
          On subsequent plays it loads the saved weights and skips training.
      5.  Type "!gen <prompt>" in chat to generate with KV-cache nucleus sampling.
         Type "!topk <prompt>" or "!nucleus <prompt>" for the other samplers.
         Type "!retrain" to wipe the saved model and retrain from scratch.
]]

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Players           = game:GetService("Players")

local RoLLM  = require(ReplicatedStorage.RoLLM)
local RoLLMt = RoLLM :: any

-- ───────────────────────── CONFIG ──────────────────────────────────────────

local STORE_KEY        = "GenerativeBot_v2"
local CORPUS_CACHE_KEY = "GenerativeBot_CorpusCache_v1"

-- GPT-2 vocab URL (replace with your own hosted vocab JSON if needed)
local _GPT2_VOCAB_URL  = ""   -- leave empty to fall back to char tokenizer (unused by default)

-- Corpus: mix of hardcoded sentences + optional HTTP fetch URLs
local CORPUS_URLS: { string } = {
	-- Example: "https://raw.githubusercontent.com/your-org/corpus/main/data.txt"
}

-- Hardcoded seed corpus (used when HTTP URLs are empty / unavailable)
local SEED_CORPUS: { string } = {
	"press space to jump over obstacles",
	"use wasd keys to move your character around the map",
	"click the shop button to buy new items and upgrades",
	"earn coins by completing quests and defeating enemies",
	"the blue portal teleports you to the next area",
	"team up with friends to defeat the final boss",
	"your health regenerates when you stand still for a moment",
	"open the inventory with the i key to manage your items",
	"the leaderboard shows the top players by score",
	"sprint by holding shift while moving forward",
	"collect gems scattered around the map for bonus points",
	"build walls and platforms to protect your base",
	"respawning takes five seconds after being defeated",
	"trade items with other players using the trade menu",
	"the red zone deals damage every second you stay in it",
	"upgrade your weapons at the blacksmith near the town square",
	"complete daily challenges to earn exclusive rewards",
	"the treasure chest respawns every ten minutes",
	"join a guild to unlock co-op missions and shared rewards",
	"your character levels up after reaching enough experience points",
	"double jump is unlocked at level five",
	"the compass arrow always points toward your active objective",
	"save your progress by visiting a checkpoint flag",
	"ranged attacks cost mana which refills over time",
	"the dragon boss appears only after all three shrines are activated",
	"type in the chat to communicate with other players",
	"you can customize your avatar in the character menu",
	"the game resets every monday morning with new weekly content",
	"find hidden rooms by looking for cracked walls and glowing tiles",
	"hold e near an npc to start a conversation",
	"use healing potions from your inventory during combat",
	"the map overview opens with the m key",
	"flying mounts are available once you reach level twenty",
	"parkour courses reward you with speed boost pickups",
	"enemy difficulty scales with the number of active players",
	"the safe zone shrinks over time in battle royale mode",
	"collect all five crystals to unlock the secret ending",
	"fishing minigame is available near any body of water",
	"pet companions provide passive stat bonuses when equipped",
	"you can report bugs and give feedback in the main menu",
	"falling off the edge respawns you at the last safe platform",
	"stealth mode lets you sneak past guards undetected",
	"the bank stores your coins safely between sessions",
	"racing mode starts every hour on the half hour",
	"using the grappling hook covers large distances quickly",
	"night cycle increases enemy spawn rate and loot quality",
	"weather effects like rain and fog change every few minutes",
	"crafting combines two materials to create a new item",
	"the tutorial island explains all the basic controls",
	"achievements unlock cosmetic rewards and title badges",
}

-- Model and training config
local MODEL_CONFIG = {
	dModel       = 128,
	numHeads     = 4,
	dFF          = 256,
	numLayers    = 4,
	maxSeqLen    = 64,
	tokenizerMode = "char",
	weightTying  = true,
}

local _MAX_ADAPTIVE_VOCAB = 512
local TRAINING_EPOCHS    = 250
local LEARNING_RATE      = 3e-4
local MAX_GEN_TOKENS     = 40

-- ───────────────────────── CORPUS ASSEMBLY ─────────────────────────────────

local function buildCorpus(): { string }
	local corpus: { string } = {}
	local seen: { [string]: boolean } = {}

	-- Always include seed corpus
	for _, s in ipairs(SEED_CORPUS) do
		if not seen[s] then seen[s] = true; table.insert(corpus, s) end
	end

	-- Optionally fetch remote corpus
	if #CORPUS_URLS > 0 then
		local fetcher = RoLLMt.CorpusFetcher.new()
		local remote  = fetcher:fetchCorpus(CORPUS_URLS, CORPUS_CACHE_KEY)
		for _, s in ipairs(remote) do
			if not seen[s] then seen[s] = true; table.insert(corpus, s) end
		end
	end

	print(string.format("[GenerativeBot] Corpus ready: %d sentences.", #corpus))
	return corpus
end

-- ───────────────────────── INITIALISE RoLLM ────────────────────────────────

print("[GenerativeBot] Building corpus …")
local corpus = buildCorpus()

print("[GenerativeBot] Initialising RoLLM …")
local llm = RoLLM.new(corpus, MODEL_CONFIG)

-- ─── Optional: wrap tokenizer with AdaptiveVocab if using BPE ──────────────
-- (Skipped here because tokenizerMode = "char"; uncomment if you switch to "bpe")
--
-- local av = RoLLMt.AdaptiveVocab.new(llm._tokenizer, _MAX_ADAPTIVE_VOCAB)
-- av:buildFromCorpus(corpus)
-- print(string.format("[GenerativeBot] AdaptiveVocab: %d dense tokens", av:getVocabSize()))

-- ───────────────────────── TRAIN OR LOAD ───────────────────────────────────

local loaded = llm:loadModel(STORE_KEY)

if loaded then
	print("[GenerativeBot] Loaded saved weights — skipping training.")
else
	print(string.format(
		"[GenerativeBot] Training for %d epochs (lr=%.4f, grad_clip=1.0, cosine LR) …",
		TRAINING_EPOCHS, LEARNING_RATE
	))

	llm:trainModel(corpus, TRAINING_EPOCHS, LEARNING_RATE, {
		batchSize        = 16,
		gradClipNorm     = 1.0,
		lrSchedule       = true,
		warmupEpochs     = 10,
		yieldEverySamples = 20,
		onEpochComplete  = function(epoch, total, avgLoss, elapsed)
			if epoch % 25 == 0 or epoch == total then
				print(string.format(
					"[GenerativeBot] Epoch %d/%d | loss=%.4f | %.1fs elapsed",
					epoch, total, avgLoss, elapsed
				))
			end
		end,
	})

	print("[GenerativeBot] Training complete. Saving model …")
	local saved = llm:saveModel(STORE_KEY)
	if saved then
		print("[GenerativeBot] Model saved.")
	else
		warn("[GenerativeBot] Save failed — weights will not persist.")
	end
end

print("[GenerativeBot] Ready. Commands: !gen <prompt>  |  !topk <prompt>  |  !nucleus <prompt>  |  !retrain")

-- ───────────────────────── CHAT HANDLER ────────────────────────────────────

local function handleChat(player: Player, message: string)
	local lower = message:lower()
	local response: string

	if lower:sub(1, 5) == "!gen " then
		local prompt = message:sub(6)
		if #prompt < 1 then return end
		local ok, res = pcall(function()
			return llm:generateWithKVCache(prompt, MAX_GEN_TOKENS, 0.9, 0.8)
		end)
		response = ok and res or ("(error: " .. tostring(res) .. ")")

	elseif lower:sub(1, 6) == "!topk " then
		local prompt = message:sub(7)
		if #prompt < 1 then return end
		local ok, res = pcall(function()
			return llm:generateTopK(prompt, MAX_GEN_TOKENS, 40, 0.9)
		end)
		response = ok and res or ("(error: " .. tostring(res) .. ")")

	elseif lower:sub(1, 9) == "!nucleus " then
		local prompt = message:sub(10)
		if #prompt < 1 then return end
		local ok, res = pcall(function()
			return llm:generateNucleus(prompt, MAX_GEN_TOKENS, 0.95, 0.85)
		end)
		response = ok and res or ("(error: " .. tostring(res) .. ")")

	elseif lower == "!retrain" then
		response = "Retraining from scratch on the next server start (delete saved weights manually via DataStore editor or bump STORE_KEY)."
	else
		return
	end

	-- Truncate to safe chat length
	if #response > 200 then
		response = response:sub(1, 197) .. "…"
	end

	-- Echo in server-side chat (visible in output + via chat service)
	print(string.format("[GenerativeBot] %s → %s", player.Name, response))
	local chatService = game:GetService("Chat")
	chatService:Chat(player.Character and player.Character:FindFirstChild("Head") or workspace, response, Enum.ChatColor.Blue)
end

local function connectPlayer(player: Player)
	player.Chatted:Connect(function(msg)
		task.spawn(handleChat, player, msg)
	end)
end

for _, player in ipairs(Players:GetPlayers()) do
	connectPlayer(player)
end
Players.PlayerAdded:Connect(connectPlayer)
