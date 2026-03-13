from . import ppat

def make(config):
    if config.model.name == "PointBERT":
        model = ppat.make(config)
    else:
        raise NotImplementedError("Model %s not supported." % config.model.name)
    return model
