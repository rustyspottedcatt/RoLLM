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
    - 40 tightly focused game sentences, all consistent vocabulary and length
    - Model: dModel=64, dFF=128, 4 heads, 2 layers (~75K params)
    - 200 epochs with batchSize=8 — drives loss low without taking hours
    - Weights saved to DataStore: training runs ONCE, then loads instantly
    - Temperature 0.2: very focused generation, close to greedy but with minor variation
    - Keyword seeding: maps player keywords to corpus-matching start phrases

    FIRST RUN:  trains for ~5-10 minutes (Studio) / ~15-20 min (in-game) then saves
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
local STORE_KEY   = "ProTalkingBot_v2"
local MAX_TOKENS  = 56
local TEMPERATURE = 0.2

-- Tightly focused corpus. Consistent vocabulary, shared prefixes, no mixed styles.
-- The more sentences share common prefixes, the easier it is for a char-level model
-- to learn the n-gram patterns that make generation feel coherent.
local CORPUS = {
	"you can move using wasd or the arrow keys",
	"you can jump by pressing the space bar",
	"you can sprint by holding the shift key",
	"you can respawn by clicking the respawn button",
	"you can see other players on the minimap",
	"you can buy upgrades from the shop menu",
	"you can collect gems to increase your score",
	"you can interact with objects by pressing e",
	"you can open the map by pressing m",
	"you can chat with other players in the chat box",
	"press space to jump over gaps and obstacles",
	"press e to open chests and pick up items",
	"press m to open the map and see the layout",
	"press tab to see the current leaderboard",
	"press shift to sprint and move faster",
	"the round starts when enough players are ready",
	"the round ends when the timer reaches zero",
	"the player with the most gems wins the round",
	"the map resets at the start of each new round",
	"the leaderboard updates after every round ends",
	"the boss appears at the end of every third round",
	"the boss drops a rare gem when defeated",
	"the golden gem is worth ten normal gems",
	"the shop sells speed boots shields and traps",
	"the minimap shows all players and gem locations",
	"gems respawn every thirty seconds across the map",
	"blue gems are worth three points each",
	"red gems are worth one point each",
	"rare gems are worth ten points each",
	"chests contain gems shields or speed boots",
	"speed boots make you faster for ten seconds",
	"a shield blocks one hit before breaking",
	"traps slow down enemies who walk into them",
	"falling off the map causes you to lose gems",
	"landing on an enemy stuns them for two seconds",
	"teamwork helps you reach higher platform areas",
	"the exit door opens only when you find the key",
	"keys are hidden inside chests around the map",
	"the timer shows how many seconds remain in the round",
	"your score is shown in the top right corner",
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
	maxSeqLen     = 45,
	tokenizerMode = "char",
})

print(string.format("[%s] %d params | vocab: %d chars | %d training samples",
	BOT_NAME, llm:getParameterCount(), llm._tokenizer:getVocabSize(), #trainingData))

-- Maps player input keywords to seed phrases that exist in the training corpus.
-- Seeds that match actual corpus prefixes give the model the best starting context.
local SEED_MAP = {
	{ keys = { "move","walk","wasd","arrow","control" }, seed = "you can move"    },
	{ keys = { "jump","space","hop","leap" },            seed = "press space"      },
	{ keys = { "sprint","fast","run","shift" },          seed = "press shift"      },
	{ keys = { "respawn","die","dead","fell" },          seed = "you can respawn"  },
	{ keys = { "gem","score","point","collect" },        seed = "you can collect"  },
	{ keys = { "shop","buy","upgrade","spend" },         seed = "you can buy"      },
	{ keys = { "map","minimap","where","location" },     seed = "the minimap"      },
	{ keys = { "round","start","begin","end","win" },    seed = "the round"        },
	{ keys = { "boss","enemy","fight","third" },         seed = "the boss"         },
	{ keys = { "chest","key","door","exit" },            seed = "the exit door"    },
	{ keys = { "shield","protect","hit","block" },       seed = "a shield"         },
	{ keys = { "speed","boot","faster" },                seed = "speed boots"      },
	{ keys = { "trap","slow","stun" },                   seed = "traps slow"       },
	{ keys = { "blue","red","rare","worth","point" },    seed = "blue gems"        },
	{ keys = { "timer","time","second","remain" },       seed = "the timer"        },
	{ keys = { "team","together","help","partner" },     seed = "teamwork"         },
	{ keys = { "leader","board","top","rank" },          seed = "the leaderboard"  },
	{ keys = { "interact","press e","open","e key" },    seed = "press e"          },
}

local function getSeed(input: string): string
	local lower = input:lower()
	for _, entry in ipairs(SEED_MAP) do
		for _, kw in ipairs(entry.keys) do
			if lower:find(kw, 1, true) then
				return entry.seed
			end
		end
	end
	-- Fall back to a random corpus phrase prefix so generation stays in-distribution
	local pick = trainingData[math.random(1, #trainingData)]
	return pick:sub(1, math.min(10, #pick))
end

local function cleanResponse(raw: string): string
	-- Trim to first sentence boundary
	local trimmed = raw:match("^(.-[%a%d][%.,!%?])")
		or raw:match("^(.+%a)")
		or raw
	trimmed = trimmed:match("^%s*(.-)%s*$") or trimmed
	if #trimmed < 5 then trimmed = raw:match("^%s*(.-)%s*$") or raw end
	-- Capitalize first letter
	return trimmed:sub(1, 1):upper() .. trimmed:sub(2)
end

-- Load or train
print(string.format("[%s] Checking DataStore for saved weights at key '%s'...", BOT_NAME, STORE_KEY))

if llm:loadModel(STORE_KEY) then
	print(string.format("[%s] Loaded saved model. Training skipped.", BOT_NAME))
else
	print(string.format(
		"[%s] No saved model found. Training %d samples for 200 epochs.",
		BOT_NAME, #trainingData))
	print(string.format("[%s] Estimated time: 5-10 min (Studio) / 15-20 min (in-game). Runs once.", BOT_NAME))

	local bestLoss    = math.huge
	local prevLoss    = math.huge
	local recentLosses: { number } = {}
	local ROLLING_N   = 10

	local history = llm:trainModel(trainingData, 200, 0.007, {
		actorCount        = 4,
		batchSize         = 8,
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

Players.PlayerAdded:Connect(function(player)
	player.Chatted:Connect(function(message)
		local lower = message:lower()
		if lower:sub(1, #CMD_PREFIX) ~= CMD_PREFIX:lower() then return end

		local input = message:sub(#CMD_PREFIX + 1):match("^%s*(.-)%s*$") or ""
		if #input == 0 then
			print(string.format("[%s] %s asked with no input — using random seed.", BOT_NAME, player.Name))
		end

		local seed = getSeed(input)
		local raw  = llm:generateTemperature(seed, MAX_TOKENS, TEMPERATURE)
		local response = cleanResponse(raw)

		print(string.format("[%s] %s asked: %q", BOT_NAME, player.Name, input))
		print(string.format("[%s] (seed: %q) -> %q", BOT_NAME, seed, response))
	end)
end)
