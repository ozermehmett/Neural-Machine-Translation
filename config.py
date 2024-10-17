class Config:
    BATCH_SIZE = 16
    MAX_LENGTH = 10
    HIDDEN_DIM = 256
    EMBEDDING_DIM = 256

    EPOCHS = 10
    LEARNING_RATE = 0.001

    SOS_token = 0
    EOS_token = 1

    MODEL_PATH = "models/weights"
    DATA_PATH = "data"
