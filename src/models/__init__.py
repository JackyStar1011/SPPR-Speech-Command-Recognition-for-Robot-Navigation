
from src.models.cnn import SpeechCommandCNN
from src.models.cnn import build_model as build_cnn_model
from src.models.cnn_gru import SpeechCommandCNNGRU
from src.models.cnn_gru import build_model as build_cnn_gru_model

__all__ = ["SpeechCommandCNN", "SpeechCommandCNNGRU", "build_model"]


def build_model(config: dict, num_classes: int):
    model_type = config["model"].get("type", "cnn_gru")
    if model_type == "cnn":
        return build_cnn_model(config, num_classes)
    if model_type == "cnn_gru":
        return build_cnn_gru_model(config, num_classes)
    raise ValueError(f"Unsupported model type: {model_type}")
