import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from datetime import datetime
from config import Config
from utils.preprocessing import load_dataset, filterPairs, prepare_data
from models.encoder import Encoder
from models.decoder import Decoder

def create_checkpoint_dir():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(Config.MODEL_PATH, f"checkpoint_{current_time}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def loss_fn(real, pred):
    """Calculate loss with masking for padding"""
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    _loss = criterion(real, pred)
    mask = tf.cast(mask, dtype=_loss.dtype)
    _loss *= mask
    return tf.reduce_mean(_loss)

def train_step(input_tensor, target_tensor, encoder, decoder,
               optimizer, enc_hidden):
    loss = 0.0
    with tf.GradientTape() as tape:
        batch_size = input_tensor.shape[0]
        enc_output, enc_hidden = encoder(input_tensor, enc_hidden)

        dec_input = tf.expand_dims([Config.SOS_token] * batch_size, 1)
        dec_hidden = enc_hidden

        for tx in range(target_tensor.shape[1]-1):
            dec_out, dec_hidden, _ = decoder(
                dec_input, dec_hidden, enc_output)
            loss += loss_fn(target_tensor[:, tx], dec_out)
            dec_input = tf.expand_dims(target_tensor[:, tx], 1)

    batch_loss = loss / target_tensor.shape[1]
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

def train():
    checkpoint_dir = create_checkpoint_dir()

    print("Loading dataset...")
    pairs = load_dataset(os.path.join(Config.DATA_PATH, "tur.txt"))
    pairs = filterPairs(pairs, Config.MAX_LENGTH)

    print("Preparing data...")
    input_tensor, target_tensor, input_lang, output_lang = prepare_data(
        pairs, 'tur', 'en', Config.MAX_LENGTH)

    print("Saving language models...")
    input_lang.save(os.path.join(Config.DATA_PATH, "input_lang.json"))
    output_lang.save(os.path.join(Config.DATA_PATH, "output_lang.json"))

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor, target_tensor)).shuffle(len(input_tensor))
    dataset = dataset.batch(Config.BATCH_SIZE, drop_remainder=True)

    print("Initializing models...")
    encoder = Encoder(input_lang.n_words, Config.HIDDEN_DIM,
                     Config.EMBEDDING_DIM, Config.BATCH_SIZE)
    decoder = Decoder(output_lang.n_words, Config.HIDDEN_DIM,
                     Config.EMBEDDING_DIM)

    optimizer = tf.keras.optimizers.Adam(Config.LEARNING_RATE)

    steps_per_epoch = len(pairs) // Config.BATCH_SIZE
    best_loss = float('inf')

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        enc_hidden = encoder.init_hidden()

        with tqdm(total=steps_per_epoch,
                 desc=f'Epoch {epoch+1}/{Config.EPOCHS}') as pbar:

            for batch, (input_tensor, target_tensor) in enumerate(
                dataset.take(steps_per_epoch)):

                batch_loss = train_step(
                    input_tensor, target_tensor, encoder, decoder,
                    optimizer, enc_hidden)

                total_loss += batch_loss
                pbar.update(1)
                pbar.set_postfix({'loss': f'{batch_loss.numpy():.4f}'})

        avg_loss = total_loss/steps_per_epoch

        if avg_loss < best_loss:
            best_loss = avg_loss
            encoder.save_weights(
                os.path.join(checkpoint_dir, f"encoder_best.h5"))
            decoder.save_weights(
                os.path.join(checkpoint_dir, f"decoder_best.h5"))

        encoder.save_weights(
            os.path.join(checkpoint_dir, f"encoder_epoch_{epoch+1}.h5"))
        decoder.save_weights(
            os.path.join(checkpoint_dir, f"decoder_epoch_{epoch+1}.h5"))

        print(f'Epoch {epoch+1} Loss {avg_loss:.4f}')

    print("Saving final models...")
    encoder.save_weights(os.path.join(Config.MODEL_PATH, "encoder.h5"))
    decoder.save_weights(os.path.join(Config.MODEL_PATH, "decoder.h5"))

    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train()
