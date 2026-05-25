--[[
    Lune unit test runner.
    Run with:  lune run tests/run_lune.luau
    Requires darklua to have already processed src/ → dist/.
    Exit code is 0 if all suites pass, 1 if any fail.
]]

local process = require("@lune/process")

local SUITES = {
	"tests/unit/LinearAlgebra.test",
	"tests/unit/Optimizer.test",
	"tests/unit/FeedForward.test",
	"tests/unit/MultiHeadAttention.test",
	"tests/unit/TransformerModel.test",
	"tests/unit/AdaptiveVocab.test",
}

local failed = 0
local passed = 0

print("═══════════════════════════════════════════════")
print(" RoLLM unit test suite")
print("═══════════════════════════════════════════════")

for _, suite in ipairs(SUITES) do
	local ok, err = pcall(require, suite)
	if ok then
		passed += 1
	else
		failed += 1
		print(string.format("✗ SUITE FAILED: %s\n  %s", suite, tostring(err)))
	end
end

print("───────────────────────────────────────────────")
print(string.format("  Suites:  %d passed, %d failed", passed, failed))
print("═══════════════════════════════════════════════")

if failed > 0 then
	process.exit(1)
end
