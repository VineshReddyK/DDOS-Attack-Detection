import numpy as np
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


def build_cnn_lstm(seq_len: int, n_features: int, num_classes: int, cfg: dict):
    from tensorflow import keras
    from tensorflow.keras import layers

    cl_cfg = cfg["cnn_lstm"]
    inputs = keras.Input(shape=(seq_len, n_features), name="sequence_input")
    x = inputs

    for filters in cl_cfg["cnn_filters"]:
        x = layers.Conv1D(filters=filters, kernel_size=cl_cfg["cnn_kernel_size"], activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2, padding="same")(x)

    for i, units in enumerate(cl_cfg["lstm_units"]):
        return_seq = i < len(cl_cfg["lstm_units"]) - 1
        x = layers.LSTM(units, return_sequences=return_seq)(x)
        x = layers.Dropout(cl_cfg["dropout_rate"])(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(cl_cfg["dropout_rate"])(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"

    model = keras.Model(inputs, outputs, name="CNN_LSTM_DDoS_Detector")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cl_cfg["learning_rate"]),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


class CNNLSTMModel:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.model = None
        self.history = None
        self.seq_len = self.config["cnn_lstm"]["sequence_length"]

    def build(self, n_features: int, num_classes: int) -> None:
        self.model = build_cnn_lstm(self.seq_len, n_features, num_classes, self.config)
        logger.info("CNN-LSTM model built: seq_len=%d, n_features=%d, num_classes=%d", self.seq_len, n_features, num_classes)

    @staticmethod
    def to_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i: i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs), np.array(ys)

    def train(self, X_train, y_train, X_val, y_val) -> None:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        cl_cfg = self.config["cnn_lstm"]
        X_train_seq, y_train_seq = self.to_sequences(X_train, y_train, self.seq_len)
        X_val_seq, y_val_seq = self.to_sequences(X_val, y_val, self.seq_len)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=cl_cfg["early_stopping_patience"],
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=cl_cfg["epochs"],
            batch_size=cl_cfg["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

    def predict(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        if y is None:
            y = np.zeros(len(X))
        X_seq, _ = self.to_sequences(X, y, self.seq_len)
        raw = self.model.predict(X_seq)
        if raw.shape[-1] == 1:
            return (raw.squeeze() > 0.5).astype(int)
        return np.argmax(raw, axis=1)

    def evaluate(self, X_test, y_test, class_names: list[str]) -> dict:
        from sklearn.metrics import classification_report, confusion_matrix

        X_seq, y_seq = self.to_sequences(X_test, y_test, self.seq_len)
        raw = self.model.predict(X_seq)
        if raw.shape[-1] == 1:
            y_pred = (raw.squeeze() > 0.5).astype(int)
        else:
            y_pred = np.argmax(raw, axis=1)

        report = classification_report(y_seq, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_seq, y_pred)
        logger.info("\n%s", classification_report(y_seq, y_pred, target_names=class_names))
        return {"classification_report": report, "confusion_matrix": cm.tolist()}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info("Saved CNN-LSTM model to %s", path)

    def load(self, path: str) -> None:
        from tensorflow import keras
        self.model = keras.models.load_model(path)
        logger.info("Loaded CNN-LSTM model from %s", path)
