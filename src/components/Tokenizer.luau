--[[
    Tokenizer Module

    @module Tokenizer
    @type TokenizerClass
]]

local CharTokenizer = require(script.Parent.CharTokenizer)
local BPETokenizer = require(script.Parent.BPETokenizer)

export type TokenizerClass = CharTokenizer.CharTokenizerClass | BPETokenizer.BPETokenizerClass

local Tokenizer = {}
Tokenizer.__index = Tokenizer

function Tokenizer.new(mode: string): TokenizerClass
    if mode == "char" then
        return CharTokenizer.new()
    elseif mode == "bpe" then
        return BPETokenizer.new()
    else
        error("Unknown tokenizer mode: " .. tostring(mode))
    end
end

return Tokenizer
