import os


class globalVal:
    device = os.environ.get("QATS_DEVICE", "cuda:0")
    loss = 0.0
    epoch = 0.0
