--[[
    _Config.luau
    
    Toggle which examples run when you start the server.
    Set an example to true to run it, false to skip it.
    
    All examples are disabled by default except ProTalkingBot.
    Enable only the ones you want to test to avoid resource contention —
    running multiple training scripts simultaneously slows all of them down
    significantly (they share the same Luau thread scheduler).
]]

return {
	TinyTraining       = false,
	ModelPersistence   = false,
	TemperatureShowcase = false,
	CorpusBuilder      = false,
	ExternalVocab      = false,
	InGameChatBot      = false,
	ProTalkingBot      = true,
}
