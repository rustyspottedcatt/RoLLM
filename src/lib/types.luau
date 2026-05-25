--[[
    Types Module

    @module Types
]]

export type TransformerConfig = {
	vocabSize: number?,
	dModel: number,
	numHeads: number,
	dFF: number,
	numLayers: number,
	maxSeqLen: number,
	tokenizerMode: string?,
	externalVocabURL: string?,
	weightTying: boolean?,
}

export type TrainingEpoch = {
	epoch: number,
	loss: number,
	averageLoss: number,
	samples: number,
}

export type TrainingOptions = {
	yieldEverySamples: number?,
	yieldEveryParameterUpdates: number?,
	onEpochComplete: ((epoch: number, totalEpochs: number, averageLoss: number, elapsedSeconds: number) -> ())?,
	batchSize: number?,
	actorCount: number?,
	gradClipNorm: number?,
	lrSchedule: boolean?,
	warmupEpochs: number?,
}

export type Matrix = { [number]: { [number]: number } }
export type Vector = { [number]: number }

export type Tokenizer = {
	getVocabSize: (self: Tokenizer) -> number,
	textToTokens: (self: Tokenizer, text: string) -> { number },
	tokensToText: (self: Tokenizer, tokens: { number }) -> string,
	buildVocabFromText: (self: Tokenizer, text: string) -> any,
	loadExternalVocab: (self: Tokenizer, url: string) -> any,
}

export type Optimizer = {
	updateParameters: (self: Optimizer, parameters: { any }, gradients: { any }, learningRate: number) -> any,
}

export type LossFunction = {
	compute: (self: LossFunction, logits: Matrix, targets: { number }) -> number,
	backward: (self: LossFunction, logits: Matrix, targets: { number }) -> Matrix,
}

return {}
