# Importação lazy para evitar carregar MLflow ao apenas importar o pacote.
__all__ = ["Predictor"]


def __getattr__(name):
    if name == "Predictor":
        from inference.predictor import Predictor
        return Predictor
    raise AttributeError(name)
