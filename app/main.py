from fastapi import FastAPI
from pydantic import BaseModel

import tensorflow as tf
from tensorflow import keras
import numpy as np

import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

# Define custom objects
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.ff_dim, activation="relu"),
                tf.keras.layers.Dense(self.embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
                "att": self.att,
                "ffn": self.ffn,
                "layernorm1": self.layernorm1,
                "layernorm2": self.layernorm2,
                "dropout1": self.dropout1,
                "dropout2": self.dropout2,
            }
        )
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=self.embed_dim, mask_zero=True
        )
        self.pos_emb = tf.keras.layers.Embedding(
            input_dim=self.maxlen, output_dim=self.embed_dim, mask_zero=True
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "maxlen": self.maxlen,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "token_emb": self.token_emb,
                "pos_emb": self.pos_emb,
            }
        )
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# load model
model = keras.models.load_model(
    "model.keras",
    custom_objects={
        "TransformerBlock": TransformerBlock,
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
    },
)


class PredData(BaseModel):
    data: list


app = FastAPI(title=f"Test-Model", version=0.1)


@app.get("/healthcheck")
def healthcheck():
    message = "API is working."
    return message


@app.post("/predict")
def predict(data: PredData):

    # Convert to array
    data = np.array(data.data)

    # Could also add logic for preprocessing
    preds = model.predict(data.reshape((1, -1)))
    label = preds.argmax(axis=1)

    return {"prediction": str(label[0])}
