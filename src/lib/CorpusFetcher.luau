--[[
    CorpusFetcher Module

    Fetches plain-text training corpora from HTTP URLs (e.g. GitHub raw files),
    splits into individual sentences, deduplicates, and optionally caches the
    raw text in a DataStore key to avoid re-fetching on subsequent training runs.

    Usage:
        local fetcher = CorpusFetcher.new()
        local sentences = fetcher:fetchCorpus({
            "https://raw.githubusercontent.com/.../corpus.txt",
        }, "MyCorpusCache_v1")    -- nil to skip caching

    Requirements:
        Studio: Game Settings > Security > Allow HTTP Requests
        Studio: Game Settings > Security > Enable Studio Access to API Services (for cache)

    @module CorpusFetcher
]]

local HttpService       = game:GetService("HttpService")
local DataStoreService  = game:GetService("DataStoreService")

local CACHE_STORE = "RoLLM_CorpusCache"

local CorpusFetcher = {}
CorpusFetcher.__index = CorpusFetcher

export type CorpusFetcherClass = {
	new: () -> CorpusFetcherClass,
	fetchCorpus: (self: CorpusFetcherClass, urls: { string }, cacheKey: string?) -> { string },
}

function CorpusFetcher.new(): CorpusFetcherClass
	local self = setmetatable({}, CorpusFetcher)
	return self
end

local function fetchText(url: string): string?
	local ok, result = pcall(function()
		return HttpService:GetAsync(url, true)
	end)
	if not ok then
		warn(string.format("[CorpusFetcher] GET failed for %s: %s", url, tostring(result)))
		return nil
	end
	return result
end

-- Split raw text into cleaned lowercase sentences.
-- Splits on newlines and sentence-ending punctuation; removes short fragments.
local function splitSentences(text: string): { string }
	local out: { string } = {}
	local seen: { [string]: boolean } = {}

	for chunk in text:gmatch("[^\n]+") do
		for sentence in (chunk .. "."):gmatch("(.-)%s*[%.!%?]") do
			local trimmed = sentence:match("^%s*(.-)%s*$") or ""
			local lower   = trimmed:lower()
			local cleaned = lower:gsub("[^%a%d%s%.,!%?'%-]", "") -- keep ascii-safe chars
			cleaned = cleaned:match("^%s*(.-)%s*$") or ""
			if #cleaned >= 8 and not seen[cleaned] then
				seen[cleaned] = true
				table.insert(out, cleaned)
			end
		end
	end

	return out
end

-- Try to load cached sentence list from DataStore.
local function loadCache(cacheKey: string): { string }?
	local ok, result = pcall(function()
		local store = DataStoreService:GetDataStore(CACHE_STORE)
		return store:GetAsync(cacheKey)
	end)
	if ok and type(result) == "table" then
		return result
	end
	return nil
end

-- Save sentence list to DataStore.  Silently skips if over the 4 MB limit.
local function saveCache(cacheKey: string, sentences: { string })
	pcall(function()
		local store = DataStoreService:GetDataStore(CACHE_STORE)
		store:SetAsync(cacheKey, sentences)
	end)
end

function CorpusFetcher:fetchCorpus(urls: { string }, cacheKey: string?): { string }
	if cacheKey then
		local cached = loadCache(cacheKey)
		if cached and #cached > 0 then
			print(string.format("[CorpusFetcher] Loaded %d sentences from cache key '%s'.", #cached, cacheKey))
			return cached
		end
	end

	local all: { string }  = {}
	local seen: { [string]: boolean } = {}

	for _, url in ipairs(urls) do
		local text = fetchText(url)
		if text then
			local sentences = splitSentences(text)
			local added = 0
			for _, s in ipairs(sentences) do
				if not seen[s] then
					seen[s] = true
					table.insert(all, s)
					added += 1
				end
			end
			print(string.format("[CorpusFetcher] %s → %d new sentences (%d total)", url, added, #all))
		end
	end

	print(string.format("[CorpusFetcher] Corpus ready: %d unique sentences.", #all))

	if cacheKey and #all > 0 then
		saveCache(cacheKey, all)
		print(string.format("[CorpusFetcher] Saved corpus to cache key '%s'.", cacheKey))
	end

	return all
end

return CorpusFetcher
