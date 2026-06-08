from ema_pytorch import EMA

def build_ema(model):
    return EMA(
        model,
        beta=0.999,
        update_every=1
    )