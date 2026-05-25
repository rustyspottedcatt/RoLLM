--[[
    Lightweight test harness for Lune unit tests.
    Import with:  local T = require("./helpers")
    Call T.report(suiteName) at the end of each test file.
    Returns false if any assertion failed (so the runner can aggregate failures).
]]

local T = {}
T._passed = 0
T._failed = 0
T._name   = "?"

-- Run `fn` as a named test case.  Failures are caught so later tests still run.
function T.test(name: string, fn: () -> ())
	local ok, err = pcall(fn)
	if ok then
		T._passed += 1
	else
		T._failed += 1
		print(string.format("  FAIL  %s\n        %s", name, tostring(err)))
	end
end

-- Basic truthy assertion
function T.ok(value: any, label: string?)
	assert(value, label or "expected truthy value")
end

-- Equality assertion (uses ==)
function T.eq(a: any, b: any, label: string?)
	local msg = label or string.format("expected %s == %s", tostring(a), tostring(b))
	assert(a == b, msg)
end

-- Near-equality for floats
function T.near(a: number, b: number, eps: number?, label: string?)
	local tol = eps or 1e-5
	local msg = label
		or string.format("expected %.8g ≈ %.8g (tol %.2g)", a, b, tol)
	assert(math.abs(a - b) <= tol, msg)
end

-- Assert that `fn` raises an error
function T.throws(fn: () -> (), label: string?)
	local ok2 = pcall(fn)
	assert(not ok2, label or "expected an error to be thrown")
end

-- Assert two 2-D matrices have the same shape
function T.shape(m: { { number } }, rows: number, cols: number, label: string?)
	local msg = label or string.format("expected shape (%d,%d)", rows, cols)
	assert(#m == rows, msg .. " (rows)")
	assert(#m[1] == cols, msg .. " (cols)")
end

-- Assert a 1-D vector has the given length
function T.len(v: { number }, n: number, label: string?)
	assert(#v == n, label or string.format("expected length %d, got %d", n, #v))
end

-- Print summary and return whether all tests passed
function T.report(suiteName: string): boolean
	local status = if T._failed == 0 then "PASS" else "FAIL"
	print(string.format("[%s] %s  %d/%d tests passed",
		suiteName, status, T._passed, T._passed + T._failed))
	return T._failed == 0
end

return T
