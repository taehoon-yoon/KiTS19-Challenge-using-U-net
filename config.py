import torch

ENCODER_CH = [1, 64, 128, 256, 512]
DECODER_CH = [512, 256, 128, 64]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PIN_MEMORY = True if DEVICE == 'cuda' else False

N_EPOCHS = 30
BATCH_SIZE = 8

INPUT_IMG_WIDTH = 512
INPUT_IMG_DEPTH = 512
