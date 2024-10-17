import gradio as gr
import tensorflow as tf
import json
from models.encoder import Encoder
from models.decoder import Decoder
from utils.language import Lang
from utils.preprocessing import normalizeString, sentencetoIndexes
from config import Config

def load_models():
    with open('data/input_lang.json', 'r', encoding='utf-8') as f:
        input_lang_dict = json.load(f)
    with open('data/output_lang.json', 'r', encoding='utf-8') as f:
        output_lang_dict = json.load(f)

    input_lang = Lang("en")
    output_lang = Lang("tr")

    input_lang.__dict__ = input_lang_dict
    output_lang.__dict__ = output_lang_dict

    encoder = Encoder(input_lang.n_words)
    decoder = Decoder(output_lang.n_words)

    encoder.load_weights('models/weights/encoder.h5')
    decoder.load_weights('models/weights/decoder.h5')

    return encoder, decoder, input_lang, output_lang

def translate_text(sentence, encoder, decoder, input_lang, output_lang, max_length=10):
    result = ''
    sentence = normalizeString(sentence)
    sentence = sentencetoIndexes(sentence, input_lang)
    sentence = tf.keras.preprocessing.sequence.pad_sequences(
        [sentence],
        padding='post',
        maxlen=max_length,
        truncating='post'
    )

    encoder_hidden = hidden = [tf.zeros((1, 256))]
    enc_out, enc_hidden = encoder(sentence, encoder_hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([Config.SOS_token], 0)

    for _ in range(max_length):
        dec_out, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
        pred = tf.argmax(dec_out, axis=1).numpy()
        word = output_lang.int2word[str(pred[0])]

        if word == "EOS":
            break

        result += word + " "
        dec_input = tf.expand_dims(pred, 0)

    return result.strip()

def create_app():
    encoder, decoder, input_lang, output_lang = load_models()

    def translate_wrapper(text):
        return translate_text(text, encoder, decoder, input_lang, output_lang)

    iface = gr.Interface(
        fn=translate_wrapper,
        inputs=gr.Textbox(
            lines=3,
            placeholder="Enter English text here...",
            label="English Text"
        ),
        outputs=gr.Textbox(
            lines=3,
            label="Turkish Translation"
        ),
        title="English to Turkish Neural Machine Translation",
        description="This application translates English text to Turkish using a neural machine translation model.",
        examples=[
            ["Hello, how are you?"],
            ["I love machine learning."],
            ["What is your name?"]
        ],
        theme=gr.themes.Soft()
    )

    return iface

def main():
    iface = create_app()
    iface.launch(share=True)

if __name__ == "__main__":
    main()
