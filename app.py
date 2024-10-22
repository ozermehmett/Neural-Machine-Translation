import gradio as gr
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from models.encoder import Encoder
from models.decoder import Decoder
from utils import Lang
from utils import normalizeString, sentencetoIndexes
from config import Config

def initialize_models(encoder, decoder):
    dummy_input = tf.zeros((1, Config.MAX_LENGTH))
    encoder_hidden = [tf.zeros((1, 256))]
    enc_out, enc_hidden = encoder(dummy_input, encoder_hidden)
    dec_input = tf.expand_dims([Config.SOS_token], 0)
    decoder(dec_input, enc_hidden, enc_out)

def load_models():
    with open('data/input_lang.json', 'r', encoding='utf-8') as f:
        input_lang_dict = json.load(f)
    with open('data/output_lang.json', 'r', encoding='utf-8') as f:
        output_lang_dict = json.load(f)

    input_lang = Lang("tr")
    output_lang = Lang("en")
    input_lang.__dict__ = input_lang_dict
    output_lang.__dict__ = output_lang_dict

    encoder = Encoder(input_lang.n_words)
    decoder = Decoder(output_lang.n_words)

    initialize_models(encoder, decoder)

    encoder.load_weights('models/weights/encoder.h5')
    decoder.load_weights('models/weights/decoder.h5')

    return encoder, decoder, input_lang, output_lang

def translate_text(sentence, encoder, decoder, input_lang, output_lang, max_length=Config.MAX_LENGTH):
    result = ''
    attention_weights_list = []

    sentence = normalizeString(sentence)
    sentence = sentencetoIndexes(sentence, input_lang)
    sentence = tf.keras.preprocessing.sequence.pad_sequences(
        [sentence],
        padding='post',
        maxlen=max_length,
        truncating='post'
    )

    encoder_hidden = [tf.zeros((1, 256))]
    enc_out, enc_hidden = encoder(sentence, encoder_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([Config.SOS_token], 0)

    for _ in range(max_length):
        dec_out, dec_hidden, attn_weights = decoder(dec_input, dec_hidden, enc_out)
        attention_weights_list.append(attn_weights)
        pred = tf.argmax(dec_out, axis=1).numpy()
        word = output_lang.int2word[str(pred[0])]

        if word == "EOS":
            break

        result += word + " "
        dec_input = tf.expand_dims(pred, 0)

    return result.strip(), tf.concat(attention_weights_list, axis=0)

def plot_attention(attention, input_sentence, predicted_sentence):
    input_sentence = normalizeString(input_sentence)
    input_tokens = input_sentence.split()
    output_tokens = predicted_sentence.split()

    fig, ax = plt.subplots(figsize=(12, 8))

    attention = attention[:len(output_tokens), :len(input_tokens)]

    im = ax.matshow(attention, cmap='RdYlBu_r', vmin=0, vmax=1)

    ax.set_xticks(range(len(input_tokens)))
    ax.set_yticks(range(len(output_tokens)))

    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels(input_tokens, rotation=45, ha='left', fontsize=12, fontweight='medium')

    ax.set_yticklabels(output_tokens, fontsize=12, fontweight='medium')

    ax.grid(True, color='gray', linestyle=':', linewidth=0.5)

    for i in range(len(output_tokens)):
        for j in range(len(input_tokens)):
            value = float(attention[i, j])
            text_color = 'black'
            text = f'{value:.2f}'
            ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.07)
    cbar.ax.set_xlabel('Attention Weight', fontsize=10)

    plt.xlabel('Turkish (Source Lang)', fontsize=12, labelpad=10)
    plt.ylabel('English (Target Lang)', fontsize=12, labelpad=10)

    plt.tight_layout()

    return fig


def create_app():
    encoder, decoder, input_lang, output_lang = load_models()

    def translate_wrapper(text):
        translation, attention_weights = translate_text(text, encoder, decoder, input_lang, output_lang)

        fig = plot_attention(attention_weights.numpy(), text, translation)

        return translation, fig

    iface = gr.Interface(
        fn=translate_wrapper,
        inputs=gr.Textbox(
            lines=3,
            placeholder="Türkçe metni buraya girin...",
            label="Türkçe Metin"
        ),
        outputs=[
            gr.Textbox(
                lines=3,
                label="English translate"
            ),
            gr.Plot(
                label="Attention Matrix Plot"
            )
        ],
        title="GRU-based Neural Machine Translation!",
        examples=[
            ["Merhaba"],
            ["Meşgül müsün"],
            ["Adın ne"],
            ["Arkadaşıma gidiyorum"]
        ],
        theme=gr.themes.Soft()
    )
    return iface

def main():
    iface = create_app()
    iface.launch(share=True)

if __name__ == "__main__":
    main()
