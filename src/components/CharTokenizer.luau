--[[
    CharTokenizer Module

    @module CharTokenizer
    @type CharTokenizerClass
]]

local CharTokenizer = {}
CharTokenizer.__index = CharTokenizer

export type CharTokenizerClass = {
    new: () -> CharTokenizerClass,
    buildVocabFromText: (self: CharTokenizerClass, text: string) -> (),
    getVocabSize: (self: CharTokenizerClass) -> number,
    textToTokens: (self: CharTokenizerClass, text: string) -> { number },
    tokensToText: (self: CharTokenizerClass, tokens: { number }) -> string,
}

function CharTokenizer.new(): CharTokenizerClass
    local self = setmetatable({}, CharTokenizer)
    self.vocab = {}
    self.ivocab = {}
    self.vocabSize = 0
    self.unknownToken = "<UNK>"
    
    return self
end

function CharTokenizer:buildVocabFromText(text: string): ()
    for c in text:gmatch(".") do
        if not self.vocab[c] then
            self.vocabSize += 1
            self.vocab[c] = self.vocabSize
            self.ivocab[self.vocabSize] = c
        end
    end
    if not self.vocab[self.unknownToken] then
        self.vocabSize += 1
        self.vocab[self.unknownToken] = self.vocabSize
        self.ivocab[self.vocabSize] = self.unknownToken
    end
end

function CharTokenizer:getVocabSize(): number
    return self.vocabSize
end

function CharTokenizer:textToTokens(text: string): { number }
    local tokens: { number } = {}
    for c in text:gmatch(".") do
        local id = self.vocab[c]
        if not id then
            id = self.vocab[self.unknownToken]
        end
        table.insert(tokens, id)
    end
    return tokens
end

function CharTokenizer:tokensToText(tokens: { number }): string
    local out: { string } = {}
    for _, id in ipairs(tokens) do
        table.insert(out, self.ivocab[id] or "?")
    end
    return table.concat(out, "")
end

return CharTokenizer
