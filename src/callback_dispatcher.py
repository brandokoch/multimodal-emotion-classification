from callbacks import MoveToGPUCallback, TrackResult

callbacks={
    'baseline_audio_model_1':[MoveToGPUCallback(), TrackResult()],
    'baseline_audio_model_2':[MoveToGPUCallback(), TrackResult()],
    'baseline_text_model':[MoveToGPUCallback(), TrackResult()],
}