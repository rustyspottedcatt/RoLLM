local T  = require("./helpers")
local LA = require("../../dist/components/LinearAlgebra")
local la = LA.new()

-- ── zeros ──────────────────────────────────────────────────────────────────
T.test("zeros: correct dimensions", function()
	local m = la:zeros(3, 5)
	T.shape(m, 3, 5)
end)

T.test("zeros: all values are 0", function()
	local m = la:zeros(2, 3)
	for i = 1, 2 do
		for j = 1, 3 do
			T.eq(m[i][j], 0)
		end
	end
end)

-- ── randomMatrix ────────────────────────────────────────────────────────────
T.test("randomMatrix: correct shape", function()
	local m = la:randomMatrix(4, 6, 0.01)
	T.shape(m, 4, 6)
end)

T.test("randomMatrix: values are small when scale is small", function()
	local m = la:randomMatrix(10, 10, 0.01)
	for i = 1, 10 do
		for j = 1, 10 do
			T.ok(math.abs(m[i][j]) < 0.1, "value unexpectedly large")
		end
	end
end)

-- ── xavierMatrix ────────────────────────────────────────────────────────────
T.test("xavierMatrix: correct shape", function()
	local m = la:xavierMatrix(8, 16)
	T.shape(m, 8, 16)
end)

T.test("xavierMatrix: mean near zero (Monte-Carlo)", function()
	local m = la:xavierMatrix(50, 50)
	local sum = 0
	for i = 1, 50 do
		for j = 1, 50 do
			sum += m[i][j]
		end
	end
	local mean = sum / (50 * 50)
	T.near(mean, 0, 0.2, "xavier mean should be near 0")
end)

-- ── mm ──────────────────────────────────────────────────────────────────────
T.test("mm: 2x2 identity pass-through", function()
	local A = { { 1, 2 }, { 3, 4 } }
	local I = { { 1, 0 }, { 0, 1 } }
	local R = la:mm(A, I)
	T.near(R[1][1], 1)
	T.near(R[1][2], 2)
	T.near(R[2][1], 3)
	T.near(R[2][2], 4)
end)

T.test("mm: 2x3 × 3x2 shape", function()
	local A = { { 1, 2, 3 }, { 4, 5, 6 } }
	local B = { { 1, 4 }, { 2, 5 }, { 3, 6 } }
	local R = la:mm(A, B)
	T.shape(R, 2, 2)
	T.near(R[1][1], 14)
	T.near(R[2][2], 77)
end)

T.test("mm: dimension mismatch throws", function()
	T.throws(function()
		la:mm({ { 1, 2 } }, { { 1, 2 } })
	end, "mm should throw on inner dimension mismatch")
end)

-- ── transpose ───────────────────────────────────────────────────────────────
T.test("transpose: shape flips", function()
	local A = { { 1, 2, 3 }, { 4, 5, 6 } }
	local At = la:transpose(A)
	T.shape(At, 3, 2)
end)

T.test("transpose: values are correct", function()
	local A = { { 1, 2 }, { 3, 4 }, { 5, 6 } }
	local At = la:transpose(A)
	T.eq(At[1][1], 1)
	T.eq(At[2][1], 3)
	T.eq(At[1][2], 5)
end)

-- ── add / subtract / scale ──────────────────────────────────────────────────
T.test("add: element-wise", function()
	local A = { { 1, 2 }, { 3, 4 } }
	local B = { { 5, 6 }, { 7, 8 } }
	local C = la:add(A, B)
	T.near(C[1][1], 6)
	T.near(C[2][2], 12)
end)

T.test("subtract: element-wise", function()
	local A = { { 5, 6 }, { 7, 8 } }
	local B = { { 1, 2 }, { 3, 4 } }
	local C = la:subtract(A, B)
	T.near(C[1][1], 4)
	T.near(C[2][2], 4)
end)

T.test("scale: scalar multiply", function()
	local A = { { 1, 2 }, { 3, 4 } }
	local R = la:scale(A, 3)
	T.near(R[1][1], 3)
	T.near(R[2][2], 12)
end)

-- ── softmax ─────────────────────────────────────────────────────────────────
T.test("softmax: rows sum to 1", function()
	local A = { { 1, 2, 3 }, { 0.5, 0.5, 0.5 } }
	local S = la:softmax(A)
	for i = 1, 2 do
		local rowSum = 0
		for j = 1, 3 do
			rowSum += S[i][j]
		end
		T.near(rowSum, 1.0, 1e-6, "row " .. i .. " must sum to 1")
	end
end)

T.test("softmax: values in (0,1)", function()
	local A = { { -100, 0, 100 } }
	local S = la:softmax(A)
	for _, v in ipairs(S[1]) do
		T.ok(v >= 0 and v <= 1)
	end
end)

-- ── relu / reluBackward ──────────────────────────────────────────────────────
T.test("relu: negative clamped to 0", function()
	T.near(la:relu(-5), 0)
	T.near(la:relu(0), 0)
	T.near(la:relu(3), 3)
end)

T.test("reluBackward: gradient gate", function()
	T.near(la:reluBackward(-1, 1), 0)
	T.near(la:reluBackward(1, 2), 2)
	T.near(la:reluBackward(0, 5), 0)
end)

print(T.report("LinearAlgebra"))
if T._failed > 0 then
	error(T._failed .. " test(s) failed in LinearAlgebra")
end
