from models.audio_model import BaselineAudioModel_1, BaselineAudioModel_2, BaselineAudioModel_3
from models.text_model import BaselineTextModel
from models.multimodal_model import MultimodalModel, MultimodalModel2

models={
    'baseline_audio_model_1': BaselineAudioModel_1(input_size=157),
    'baseline_audio_model_2': BaselineAudioModel_2(),
    'baseline_audio_model_3': BaselineAudioModel_3(),
    'baseline_text_model': BaselineTextModel(),
    'multimodal_model': MultimodalModel(),
    'multimodal_model2': MultimodalModel2()
}