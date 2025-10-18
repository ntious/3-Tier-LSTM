from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, regularizers, models

def build_hybrid_3tier_lstm(
    input_dim: int,
    tier1_dense_units: int = 64,
    lstm_units: int = 64,
    tier3_dense_units: int = 32,
    dropout: float = 0.2,
    l2_val: float = 1e-4
) -> tf.keras.Model:
    """
    Hybrid 3-Tier architecture:
    Tier 1: Dense feature extraction
    Tier 2: LSTM to capture interactions (timesteps=1)
    Tier 3: Dense prediction head
    """
    inputs = layers.Input(shape=(1, input_dim), name="inputs")  # (batch, timesteps=1, features)
    x = layers.TimeDistributed(layers.Dense(tier1_dense_units, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_val)))(inputs)
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(tier3_dense_units, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_val))(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="Hybrid3TierLSTM")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
