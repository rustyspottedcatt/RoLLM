--[[
    CrossEntropyLoss Module

    @module CrossEntropyLoss
    @type CrossEntropyLossClass
]]
local CrossEntropyLoss = {}
CrossEntropyLoss.__index = CrossEntropyLoss

export type CrossEntropyLossClass = {
    new: () -> CrossEntropyLossClass,
    compute: (self: CrossEntropyLossClass, logits: { { number } }, targets: { number }) -> number,
    backward: (self: CrossEntropyLossClass, logits: { { number } }, targets: { number }) -> { { number } },
}

function CrossEntropyLoss.new(): CrossEntropyLossClass
    local self = setmetatable({}, CrossEntropyLoss)
    self.loss = 0
    self.probs = {}
    self.grad = {} 
    return self
end

local function softmax(logits: { number }): { number }
    local maxLogit = -math.huge

    for _, logit in ipairs(logits) do
        if logit > maxLogit then
            maxLogit = logit
        end
    end

    local sumExp = 0
    local probs = {}

    for _, logit in ipairs(logits) do
        local expVal = math.exp(logit - maxLogit)
        table.insert(probs, expVal)
        sumExp += expVal
    end

    for i, prob in ipairs(probs) do
        probs[i] = prob / sumExp
    end

    return probs
end

function CrossEntropyLoss:compute(logits: { { number } }, targets: { number }): number
    self.loss = 0
    self.probs = {}

    for i, logitRow in ipairs(logits) do
        local probs = softmax(logitRow)
        self.probs[i] = probs 

        local target = targets[i]

        if target == nil then
            error(string.format("Target at index %d is nil. Logits: %s", i, table.concat(logitRow, ", ")))
        end
        if type(target) ~= "number" then
            error(string.format("Target at index %d is not a number. Logits: %s", i, table.concat(logitRow, ", ")))
        end
        if target < 1 or target > #logitRow then
            error(string.format("Target at index %d is out of range. Target: %d, Logit Row Size: %d", i, target, #logitRow))
        end

        local targetProb = probs[target]

        if targetProb <= 0 then
            error(string.format("Probability at index %d for target %d is not positive. Probability: %f", i, target, targetProb))
        end

        self.loss += -math.log(targetProb)
    end

    return self.loss
end

function CrossEntropyLoss:backward(logits: { { number } }, targets: { number }): { { number } }
    self.grad = {}

    for i, logitRow in ipairs(logits) do
        self.grad[i] = {}
        local probs = self.probs[i]
        local target = targets[i]

        for j, prob in ipairs(probs) do
            if j == target then
                self.grad[i][j] = prob - 1
            else
                self.grad[i][j] = prob
            end
        end
    end

    return self.grad
end

return CrossEntropyLoss
