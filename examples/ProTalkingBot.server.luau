--[[
    ProTalkingBot.server.luau

    The highest-quality talking model achievable inside Roblox without external APIs.

    WHY NOT GPT-2 VOCAB?
    The GPT-2 vocabulary is just a 50,257-token dictionary. Coherent output comes
    from the WEIGHTS, not the vocab. Loading GPT-2's actual pretrained weights
    requires ~500MB of data — not feasible in Roblox. With random weights and a
    50,257-token output softmax, the model needs exponentially more training data
    and time to converge compared to a focused char-level setup.

    WHAT THIS DOES INSTEAD:
    - 100 game-help phrases across 10 topics (movement, combat, items, economy, etc.)
    - Model: dModel=64, dFF=128, 4 heads, 2 layers (~70K params)
    - 300 epochs with batchSize=16 — sufficient for 100-phrase memorisation
    - Weights saved to DataStore: training runs ONCE, then loads instantly
    - Temperature 0.2: focused generation, close to greedy
    - Word-overlap routing: content words in player input are matched against corpus
      phrases; no hardcoded keyword lists — scales with the corpus automatically

    FIRST RUN:  trains for ~20-30 minutes (Studio) / ~40 min (in-game) then saves
    LATER RUNS: loads instantly, skips training entirely

    Command: !talk <your message>

    Requirements:
      Studio: Game Settings > Security > Enable Studio Access to API Services
]]

local Config = require(script.Parent._Config)
if not Config[script.Name] then return end

local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RoLLM = require(ReplicatedStorage.RoLLM)

local BOT_NAME    = "ProBot"
local CMD_PREFIX  = "!talk"
local STORE_KEY   = "ProTalkingBot_v3"
local MAX_TOKENS  = 50
local TEMPERATURE = 0.2

-- 100-phrase corpus covering movement, controls, items, combat, economy, map,
-- game modes, social, tips, and misc. More diverse coverage means word-overlap
-- seed routing can handle most natural player questions without a hardcoded map.
local CORPUS = {
	-- Movement
	"you can move using wasd or the arrow keys",
	"you can jump by pressing the space bar",
	"you can sprint by holding the shift key",
	"you can double jump by pressing space twice",
	"you can swim by moving into the water",
	"you can climb ladders by walking into them",
	"you can crouch by pressing the ctrl key",
	"you can dash by double tapping a direction key",
	"you can slow down by releasing all movement keys",
	"you can fly in creative mode with the f key",
	-- Controls
	"press e to interact with objects and npcs",
	"press m to open and close the map",
	"press tab to view the leaderboard scores",
	"press esc to open the settings and pause menu",
	"press g to drop the item you are holding",
	"press r to reload your weapon when ammo is low",
	"press q to swap between your two equipped items",
	"press f to pick up items and gems from the ground",
	"press t to open the team chat for your group",
	"press b to open the backpack and view your items",
	-- Items
	"gems are the main currency used in this game",
	"blue gems are worth three points each",
	"red gems are worth one point each",
	"rare gems are worth ten points each",
	"golden gems are worth fifty points each",
	"chests contain gems shields or speed boots",
	"keys are hidden inside chests around the map",
	"speed boots make you faster for ten seconds",
	"a shield blocks one hit before it breaks",
	"traps slow down any player who steps on them",
	-- Combat
	"you can attack by clicking the left mouse button",
	"you can block attacks by holding right mouse button",
	"you can dodge by pressing shift while moving",
	"landing on an enemy stuns them for two seconds",
	"falling off the map causes you to lose five gems",
	"you can heal by standing still for three seconds",
	"a stunned enemy cannot move or attack briefly",
	"your health bar is shown in the bottom left corner",
	"you respawn at the nearest checkpoint after dying",
	"hitting a trap deals double damage to the enemy",
	-- Economy
	"you can buy upgrades from the shop in the town",
	"the shop sells speed boots shields and traps",
	"upgrades cost gems and last until the round ends",
	"selling items at the shop gives you bonus gems",
	"the shop refreshes its stock at the start of rounds",
	"rare items in the shop appear only every third round",
	"you can spend gems to unlock new character skins",
	"daily quests reward you with bonus gems when done",
	"you earn extra gems by finishing in the top three",
	"the premium shop contains exclusive cosmetic items",
	-- Map and navigation
	"the minimap shows all players and gem locations",
	"waypoints appear on the minimap when a quest is active",
	"the exit door opens only when you find the key",
	"green zones on the map are safe areas to rest",
	"red zones on the map have stronger enemies and loot",
	"you can see other players as dots on the minimap",
	"the map resets at the start of each new round",
	"hidden areas on the map contain rare gems and chests",
	"the spawn point is marked with a blue star on the map",
	"portals on the map teleport you to different zones",
	-- Game modes
	"the round starts when enough players are ready",
	"the round ends when the timer reaches zero seconds",
	"the player with the most gems wins the round",
	"the leaderboard updates after every round ends",
	"the boss appears at the end of every third round",
	"the boss drops a rare gem when defeated by a player",
	"the timer shows how many seconds remain in the round",
	"gems respawn every thirty seconds across the map",
	"bonus rounds give double gem rewards for one minute",
	"survival mode ends when only one player remains alive",
	-- Social
	"you can chat with other players in the chat box",
	"you can form a team with up to four other players",
	"teamwork helps you reach higher platform areas faster",
	"you can trade gems with teammates during a round",
	"the team with the most combined gems wins team mode",
	"you can invite friends to your team from the menu",
	"your score is shown in the top right corner of screen",
	"you can see other players health bars above their heads",
	"sending a friend request adds them to your friends list",
	"you can mute other players in the settings menu",
	-- Tips
	"collect gems quickly before other players take them",
	"use speed boots to escape from enemies chasing you",
	"place traps near gem spawns to catch other players",
	"save your shield for the boss fight at round three",
	"farm blue gems early and spend on upgrades midgame",
	"the golden gem spawns once per round near the center",
	"stay near teammates when exploring dangerous red zones",
	"use the minimap to track where other players are moving",
	"finishing a quest gives more gems than farming alone",
	"practice the double jump to reach secret platform areas",
	-- Misc
	"you can respawn by clicking the respawn button on screen",
	"your settings are saved automatically when you leave",
	"the tutorial teaches you the basic controls and rules",
	"your progress is saved to the server every ten seconds",
	"the daily login reward gives you bonus gems each day",
	"you can view your stats in the profile menu",
	"achievements reward you with gems and exclusive badges",
	"contacting support can be done from the settings menu",
	"you can report players using the report button in menu",
	"the exit door opens after you collect the golden key",
}

local function buildTrainingData(raw: { string }): { string }
	local seen: { [string]: boolean } = {}
	local out: { string } = {}
	for _, s in ipairs(raw) do
		local clean = (s:lower():match("^%s*(.-)%s*$") or s)
		if #clean >= 8 and not seen[clean] then
			seen[clean] = true
			table.insert(out, clean)
		end
	end
	return out
end

local trainingData = buildTrainingData(CORPUS)

local llm = RoLLM.new(trainingData, {
	dModel        = 64,
	numHeads      = 4,
	dFF           = 128,
	numLayers     = 2,
	maxSeqLen     = 60,
	tokenizerMode = "char",
})

print(string.format("[%s] %d params | vocab: %d chars | %d training samples",
	BOT_NAME, llm:getParameterCount(), llm._tokenizer:getVocabSize(), #trainingData))

-- Common English stopwords — excluded from overlap scoring so only content words match.
local STOPWORDS: { [string]: boolean } = {
	["the"]=true, ["a"]=true, ["an"]=true, ["is"]=true, ["it"]=true, ["in"]=true,
	["on"]=true, ["at"]=true, ["to"]=true, ["do"]=true, ["i"]=true, ["me"]=true,
	["my"]=true, ["we"]=true, ["you"]=true, ["he"]=true, ["she"]=true, ["are"]=true,
	["was"]=true, ["be"]=true, ["by"]=true, ["or"]=true, ["and"]=true, ["of"]=true,
	["for"]=true, ["can"]=true, ["how"]=true, ["what"]=true, ["when"]=true,
	["where"]=true, ["who"]=true, ["why"]=true, ["this"]=true, ["that"]=true,
	["with"]=true, ["from"]=true, ["up"]=true, ["out"]=true,
}

-- Word-overlap seed selection: score every corpus phrase by how many content words
-- it shares with the player's input, then use the first 3 words of the winner as seed.
-- Scales automatically as the corpus grows — no hardcoded keyword lists needed.
local function getBestSeed(input: string): string
	local inputWords: { [string]: boolean } = {}
	for word in input:lower():gmatch("%a+") do
		if not STOPWORDS[word] then
			inputWords[word] = true
		end
	end

	local bestPhrase: string? = nil
	local bestScore  = 0
	for _, phrase in ipairs(trainingData) do
		local score = 0
		for word in phrase:gmatch("%a+") do
			if inputWords[word] then score += 1 end
		end
		if score > bestScore then
			bestScore = score
			bestPhrase = phrase
		end
	end

	if bestPhrase and bestScore > 0 then
		local words: { string } = {}
		for w in bestPhrase:gmatch("%S+") do
			table.insert(words, w)
			if #words >= 3 then break end
		end
		return table.concat(words, " ")
	end

	-- No corpus overlap: seed from the player's own words (first 20 chars lowercase)
	local raw = input:lower():match("^%s*(.-)%s*$") or ""
	if #raw >= 2 then return raw:sub(1, math.min(20, #raw)) end

	-- Last resort: random corpus phrase prefix
	local pick = trainingData[math.random(1, #trainingData)]
	return pick:sub(1, math.min(12, #pick))
end

local function cleanResponse(raw: string): string
	local lower = raw:lower()

	-- 1. Corpus-prefix match: find the longest known phrase the output begins with,
	--    then truncate to exactly that phrase (eliminates post-phrase garbage entirely).
	local bestLen = 0
	for _, phrase in ipairs(trainingData) do
		if #phrase > bestLen and lower:sub(1, #phrase) == phrase then
			bestLen = #phrase
		end
	end
	if bestLen >= 5 then
		local out = raw:sub(1, bestLen)
		return out:sub(1, 1):upper() .. out:sub(2)
	end

	-- 2. Heuristic fallback for unmatched seeds: detect garbage (4+ identical
	--    consecutive chars like "oooo") and trim to the last clean word before it.
	local clean = raw
	local garbled = clean:find("(.)%1%1%1")
	if garbled then
		local lastSpace = clean:sub(1, garbled - 1):match(".*%s()")
		clean = clean:sub(1, (lastSpace or garbled) - 1)
	end
	clean = clean:match("^(.+%a)") or clean
	clean = clean:match("^%s*(.-)%s*$") or clean
	if #clean < 3 then clean = raw:match("^%s*(.-)%s*$") or raw end
	return clean:sub(1, 1):upper() .. clean:sub(2)
end

-- Load or train
print(string.format("[%s] Checking DataStore for saved weights at key '%s'...", BOT_NAME, STORE_KEY))

if llm:loadModel(STORE_KEY) then
	print(string.format("[%s] Loaded saved model. Training skipped.", BOT_NAME))
else
	print(string.format(
		"[%s] No saved model found. Training %d samples for 300 epochs.",
		BOT_NAME, #trainingData))
	print(string.format("[%s] Estimated time: 5-10 min (Studio) / 15-20 min (in-game). Runs once.", BOT_NAME))

	local bestLoss    = math.huge
	local prevLoss    = math.huge
	local recentLosses: { number } = {}
	local ROLLING_N   = 10

	local history = llm:trainModel(trainingData, 300, 0.007, {
		actorCount        = 4,
		batchSize         = 16,
		yieldEverySamples = 1,
		onEpochComplete   = function(epoch, total, avgLoss, elapsed)
			-- Track best and rolling average
			if avgLoss < bestLoss then bestLoss = avgLoss end
			table.insert(recentLosses, avgLoss)
			if #recentLosses > ROLLING_N then table.remove(recentLosses, 1) end
			local rollingSum = 0
			for _, v in ipairs(recentLosses) do rollingSum += v end
			local rollingAvg = rollingSum / #recentLosses

			local delta   = prevLoss - avgLoss
			local trend   = if delta > 0.01 then "v" elseif delta < -0.01 then "^" else "~"
			prevLoss = avgLoss

			local throughput = (#trainingData * epoch) / math.max(elapsed, 0.001)

			if epoch % 10 == 0 or epoch == total or epoch == 1 then
				local pct = math.floor((epoch / total) * 30)
				local bar = string.rep("|", pct) .. string.rep(".", 30 - pct)
				local eta = if epoch < total then (elapsed / epoch) * (total - epoch) else 0
				local etaM = math.floor(eta / 60)
				local etaS = math.floor(eta % 60)
				local elM  = math.floor(elapsed / 60)
				local elS  = math.floor(elapsed % 60)

				print(string.format(
					"[%s] %3d/%d [%s] %s\n" ..
					"       Loss: %.4f (best: %.4f, delta: %+.4f)\n" ..
					"       Roll: %.4f (last %d)  |  Throughput: %.0f samples/s\n" ..
					"       Time: %dm%02ds elapsed  |  ETA: %dm%02ds",
					BOT_NAME, epoch, total, bar, trend,
					avgLoss, bestLoss, delta,
					rollingAvg, #recentLosses, throughput,
					elM, elS, etaM, etaS
				))
			end
		end,
	})

	local finalLoss = history[#history].averageLoss
	print(string.format("[%s] Training complete. Final loss: %.4f", BOT_NAME, finalLoss))

	if finalLoss > 8 then
		warn(string.format(
			"[%s] Loss is still high (%.4f). Responses may be incoherent. "
			.. "Consider more epochs or a smaller model.", BOT_NAME, finalLoss))
	end

	if llm:saveModel(STORE_KEY) then
		print(string.format("[%s] Weights saved to DataStore. Future runs load instantly.", BOT_NAME))
	else
		warn(string.format(
			"[%s] DataStore save failed. Check that API Services are enabled in Studio.",
			BOT_NAME))
	end
end

-- Self-test: generate from a handful of seeds and print to verify quality
print(string.format("\n[%s] Self-test generations (temp=%.1f):", BOT_NAME, TEMPERATURE))
local testSeeds = { "you can move", "press space", "the round", "the boss", "speed boots" }
for _, seed in ipairs(testSeeds) do
	local out = cleanResponse(llm:generateTemperature(seed, MAX_TOKENS, TEMPERATURE))
	print(string.format("  %q -> %q", seed, out))
end
print("")

print(string.format("[%s] Ready. Type '%s <message>' to talk.", BOT_NAME, CMD_PREFIX))

local function handleChat(player: Player, message: string)
	local lower = message:lower()
	if lower:sub(1, #CMD_PREFIX) ~= CMD_PREFIX:lower() then return end

	local input = message:sub(#CMD_PREFIX + 1):match("^%s*(.-)%s*$") or ""
	if #input == 0 then
		print(string.format("[%s] %s asked with no input — using random seed.", BOT_NAME, player.Name))
	end

	local seed     = getBestSeed(input)
	local raw      = llm:generateTemperature(seed, MAX_TOKENS, TEMPERATURE)
	local response = cleanResponse(raw)

	print(string.format("[%s] %s asked: %q", BOT_NAME, player.Name, input))
	print(string.format("[%s] (seed: %q) -> %q", BOT_NAME, seed, response))
end

-- Connect players already in the game (e.g. Studio tester who joined before training finished)
for _, player in ipairs(Players:GetPlayers()) do
	player.Chatted:Connect(function(msg) handleChat(player, msg) end)
end

-- Connect future joiners
Players.PlayerAdded:Connect(function(player)
	player.Chatted:Connect(function(msg) handleChat(player, msg) end)
end)
