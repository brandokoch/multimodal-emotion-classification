from callbacks import MoveToGPUCallback, TrackResult, MultiModalMoveToGPUCallback

callbacks={
    'default':[MoveToGPUCallback(), TrackResult()],
    'multimodal':[MultiModalMoveToGPUCallback(), TrackResult()]
}