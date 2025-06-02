--[[
    LinearAlgebra Module

    @module LinearAlgebra
    @type LinearAlgebraClass
]]

local Types = require(script.Parent.Parent.lib.types)

export type Matrix = Types.Matrix
export type Vector = Types.Vector

export type LinearAlgebraClass = {
	new: () -> LinearAlgebraClass,
	zeros: (self: LinearAlgebraClass, rows: number, cols: number) -> Matrix,
	randomMatrix: (self: LinearAlgebraClass, rows: number, cols: number, scale: number?) -> Matrix,
	mm: (self: LinearAlgebraClass, A: Matrix, B: Matrix) -> Matrix,
	add: (self: LinearAlgebraClass, A: Matrix, B: Matrix) -> Matrix,
	subtract: (self: LinearAlgebraClass, A: Matrix, B: Matrix) -> Matrix,
	scale: (self: LinearAlgebraClass, A: Matrix, s: number) -> Matrix,
	transpose: (self: LinearAlgebraClass, A: Matrix) -> Matrix,
	softmax: (self: LinearAlgebraClass, A: Matrix) -> Matrix,
	elementwiseMultiply: (self: LinearAlgebraClass, A: Matrix, B: Matrix) -> Matrix,
	elementwiseAdd: (self: LinearAlgebraClass, A: Matrix, B: Matrix) -> Matrix,
	elementwiseSubtract: (self: LinearAlgebraClass, A: Matrix, B: Matrix) -> Matrix,
	relu: (self: LinearAlgebraClass, A: Matrix) -> Matrix,
	reluBackward: (self: LinearAlgebraClass, A: Matrix, gradOutput: Matrix) -> Matrix,
}

local LinearAlgebra = {}
LinearAlgebra.__index = LinearAlgebra

function LinearAlgebra.new(): LinearAlgebraClass
	local self = setmetatable({}, LinearAlgebra)
	return self
end

function LinearAlgebra:zeros(rows: number, cols: number): Matrix
	if rows < 1 or cols < 1 then
		error("zeros: rows and cols must be ≥ 1")
	end

	local mat: Matrix = {}
	for i = 1, rows do
		local row = {}
		for j = 1, cols do
			row[j] = 0
		end
		mat[i] = row
	end

	return mat
end

function LinearAlgebra:randomMatrix(rows: number, cols: number, scale: number?): Matrix
	local mat: Matrix = {}
	for i = 1, rows do
		local row = {}
		for j = 1, cols do
			row[j] = (math.random() * 2 - 1) * (scale or 0.1)
		end
		mat[i] = row
	end
	return mat
end

function LinearAlgebra:mm(A: Matrix, B: Matrix): Matrix
	local m = #A
	local n = #A[1]
	local p = #B[1]
	local C = self:zeros(m, p)
	local B_T = self:transpose(B)
	for i = 1, m do
		for j = 1, p do
			local sum = 0
			local rowA = A[i]
			local rowB_T = B_T[j]
			for k = 1, n do
				sum += rowA[k] * rowB_T[k]
			end
			C[i][j] = sum
		end
	end
	return C
end

function LinearAlgebra.mmAsync(self, A: { { number } }, B: { { number } }, callback: (result: { { number } }) -> ()): ()
	coroutine.wrap(function()
		local result = {}
		local A_rows = #A
		local A_cols = #A[1]
		local B_cols = #B[1]

		local B_t = {}
		for i = 1, #B do
			for j = 1, #B[1] do
				B_t[j] = B_t[j] or {}
				B_t[j][i] = B[i][j]
			end
		end

		for i = 1, A_rows do
			result[i] = {}
			for j = 1, B_cols do
				local sum = 0
				local rowA = A[i]
				local rowB_t = B_t[j]
				for k = 1, A_cols do
					sum = sum + rowA[k] * rowB_t[k]
				end
				result[i][j] = sum
			end
			coroutine.yield()
		end

		callback(result)
	end)()
end

function LinearAlgebra:add(A: Matrix, B: Matrix): Matrix
	local rows = #A
	local cols = #A[1]
	local C = self:zeros(rows, cols)
	for i = 1, rows do
		for j = 1, cols do
			C[i][j] = A[i][j] + B[i][j]
		end
	end
	return C
end

function LinearAlgebra:subtract(A: Matrix, B: Matrix): Matrix
	local rows = #A
	local cols = #A[1]
	local C = self:zeros(rows, cols)
	for i = 1, rows do
		for j = 1, cols do
			C[i][j] = A[i][j] - B[i][j]
		end
	end
	return C
end

function LinearAlgebra:scale(A: Matrix, s: number): Matrix
	local rows = #A
	local cols = #A[1]
	local C = self:zeros(rows, cols)
	for i = 1, rows do
		for j = 1, cols do
			C[i][j] = A[i][j] * s
		end
	end
	return C
end

function LinearAlgebra:transpose(A: Matrix): Matrix
	local rows = #A
	local cols = #A[1]
	local T = self:zeros(cols, rows)
	for i = 1, rows do
		for j = 1, cols do
			T[j][i] = A[i][j]
		end
	end
	return T
end

function LinearAlgebra:softmax(A: Matrix): Matrix
	local rows = #A
	local cols = #A[1]
	local out = self:zeros(rows, cols)

	for i = 1, rows do
		local rowMax = -math.huge
		for j = 1, cols do
			if A[i][j] > rowMax then
				rowMax = A[i][j]
			end
		end

		local sumExp = 0
		for j = 1, cols do
			local x = A[i][j] - rowMax
			if x > 700 then
				x = 700
			elseif x < -700 then
				x = -700
			end
			out[i][j] = math.exp(x)
			sumExp += out[i][j]
		end

		for j = 1, cols do
			out[i][j] = out[i][j] / sumExp
		end
	end

	return out
end

function LinearAlgebra:elementwiseMultiply(A: Matrix, B: Matrix): Matrix
	local rows = #A
	local cols = #A[1]
	local C = self:zeros(rows, cols)
	for i = 1, rows do
		for j = 1, cols do
			C[i][j] = A[i][j] * B[i][j]
		end
	end
	return C
end

function LinearAlgebra:elementwiseAdd(A: Matrix, B: Matrix): Matrix
	assert(#A == #B and #A[1] == #B[1], "Matrix dimensions must match")
	return self:add(A, B)
end

function LinearAlgebra:elementwiseSubtract(A: Matrix, B: Matrix): Matrix
	assert(#A == #B and #A[1] == #B[1], "Matrix dimensions must match")
	return self:subtract(A, B)
end

function LinearAlgebra:relu(A: Matrix): Matrix
	local rows = #A
	local cols = #A[1]
	local C = self:zeros(rows, cols)
	for i = 1, rows do
		for j = 1, cols do
			C[i][j] = math.max(0, A[i][j])
		end
	end
	return C
end

function LinearAlgebra:reluBackward(A: Matrix, gradOutput: Matrix): Matrix
	local rows = #A
	local cols = #A[1]
	local gradInput = self:zeros(rows, cols)
	for i = 1, rows do
		for j = 1, cols do
			gradInput[i][j] = A[i][j] > 0 and gradOutput[i][j] or 0
		end
	end
	return gradInput
end

return LinearAlgebra
