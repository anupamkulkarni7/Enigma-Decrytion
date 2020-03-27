import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.layers import Input, Embedding, Bidirectional, TimeDistributed, LSTM, Dense

from utils import *


class Config:
    def __init__(self, epochs=10, embed_dim=10, n_hidden=50, mask_zero=True, max_len=42, batch_size=32):
        self.mask_zero = mask_zero
        self.inp_max_len = max_len
        self.embed_dim = embed_dim
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.batch_size = batch_size


class CharTokenizer:
    def __init__(self):
        self.char_idx = {}
        self.idx_char = {}
        self.vocab_size = None

    def fit_to_text(self, texts):
        # Input is a list of strings
        for text in texts:
            for char in list(text):
                if char not in self.char_idx:
                    key = len(self.char_idx) + 1
                    self.char_idx[char] = key
                    self.idx_char[key] = char

        self.vocab_size = len(self.char_idx) + 1

    def text_to_sequence(self, texts):
        # Input is a list of strings
        sequence = []
        for text in texts:
            row = [self.char_idx[c] for c in list(text)]
            sequence.append(row)
        return sequence

    def sequence_to_text(self, seq):
        texts = []
        for entry in seq:
            row = [self.idx_char[c] for c in entry]
            text = ''.join(row)
            texts.append(text)
        return texts


def initialize_tokenizer(texts):
    tokenizer = CharTokenizer()
    tokenizer.fit_to_text(texts)
    return tokenizer


def transform_to_numeric(tokenizer_x, texts_x, tokenizer_y, texts_y, cfg):

    max_len = cfg.inp_max_len
    padding_type = "post"
    trunc_type = "post"

    char_indices_x = tokenizer_x.text_to_sequence(texts_x)
    padded_seq_x = pad_sequences(char_indices_x, padding=padding_type, maxlen=max_len, truncating=trunc_type)

    char_indices_y = tokenizer_y.text_to_sequence(texts_y)
    padded_seq_y = pad_sequences(char_indices_y, padding=padding_type, maxlen=max_len, truncating=trunc_type)

    return padded_seq_x, padded_seq_y


def define_model(tokenizer, cfg, n_y):
    """
        Args:
            tokenizer: input sequence tokenizer
            cfg: Config obj containing the model parameters
            n_y: (int) Number of characters in the decrypted seq (26)
        """
    vocab_size = tokenizer.vocab_size
    output_dim = cfg.embed_dim
    mask_zero = cfg.mask_zero
    max_len = cfg.inp_max_len
    n_hidden = cfg.n_hidden

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=output_dim,
                                trainable=True, mask_zero=mask_zero)
    input_shape = (max_len,)
    char_indices = Input(shape=input_shape, dtype='int32')
    embeddings = embedding_layer(char_indices)

    lstm_seq = Bidirectional(LSTM(n_hidden, return_sequences=True, return_state=False))(embeddings)

    y_pred = TimeDistributed(Dense(n_y, activation="softmax"))(lstm_seq)

    model = Model(inputs=char_indices, outputs=y_pred)

    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def save_decrypt_model(model, tokenizer_x, tokenizer_y, cfg,
                       tokenizer_file="tokenizer.pkl",
                       model_file="decrypt_model.h5"):

    print("Saving model weights to files...\nTokenizer: {}\nJoint weights: {}".
          format(tokenizer_file, model_file))

    pkl_dump((tokenizer_x, tokenizer_y, cfg), tokenizer_file)
    model.save(model_file)


def load_decrypt_model(tokenizer_file="tokenizer.pkl",
                       model_file="decrypt_model.h5"):

    print("Loading model weights to files...\nTokenizer: {}\nJoint weights: {}".
          format(tokenizer_file, model_file))
    tokenizer_x, tokenizer_y, cfg = pkl_load(tokenizer_file)
    model = load_model(model_file)
    return tokenizer_x, tokenizer_y, cfg, model


def train_model(train_x, train_y, test_x=None, test_y=None,
                tokenizer_file="tokenizer.pkl",
                model_file="decrypt_model.h5"):

    print("Building tokenizers..")
    tokenizer_x = initialize_tokenizer(train_x)
    tokenizer_y = initialize_tokenizer(train_y)
    cfg = Config()
    nd_chars = len(tokenizer_y.char_idx) + 1

    print("Transforming data to numeric..")
    train_seq_x, train_seq_y = transform_to_numeric(tokenizer_x, train_x, tokenizer_y, train_y, cfg)

    print("Defining model..")
    model = define_model(tokenizer_x, cfg, n_y=nd_chars)

    if True:
        print("Training model..")
        history = model.fit(train_seq_x, train_seq_y,
                            epochs=cfg.epochs, batch_size=cfg.batch_size, shuffle=True)

        if test_x and test_y:
            print("Evaluating model on test set..")
            test_seq_x, test_seq_y = transform_to_numeric(tokenizer_x, test_x, tokenizer_y, test_y, cfg)
            model.evaluate(test_seq_x, test_seq_y)

        save_decrypt_model(model, tokenizer_x, tokenizer_y, cfg, tokenizer_file, model_file)


def test_model(test_x, test_y, tokenizer_file="tokenizer.pkl",
               model_file="decrypt_model.h5"):

    tokenizer_x, tokenizer_y, cfg, model = load_decrypt_model(tokenizer_file, model_file)
    test_seq_x, test_seq_y = transform_to_numeric(tokenizer_x, test_x, tokenizer_y, test_y, cfg)
    model.evaluate(test_seq_x, test_seq_y)

    p_y = model.predict(test_seq_x)
    # print(np.shape(p_y))

    p_y = np.argmax(p_y, axis=-1)
    p_yl = list(p_y)

    if True:
        y_pred_i = []
        for y_p, txt in zip(p_yl, test_x):
            y_pn = list(y_p[:len(txt)])
            y_pred_i.append(y_pn)

        y_pred = tokenizer_y.sequence_to_text(y_pred_i)
        print("Test Example:")
        print("Encrypted seq: {} \nCorrect decryption: {}\nPredicted decryption: {}"
              .format(test_x[0], test_y[0], y_pred[0]))
        return y_pred


