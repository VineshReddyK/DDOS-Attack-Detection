import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def build_ann(input_dim, num_classes, cfg):
    # keras is imported inside the function on purpose — tensorflow is slow to
    # import, and nothing here needs it until we actually build a model.
    from tensorflow import keras
    from tensorflow.keras import layers

    ann_cfg = cfg["ann"]
    model = keras.Sequential(name="ANN_DDoS_Detector")
    model.add(layers.Input(shape=(input_dim,)))

    # one Dense -> BatchNorm -> Dropout block per hidden layer in the config
    for units in ann_cfg["hidden_layers"]:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(ann_cfg["dropout_rate"]))

    # binary uses a single sigmoid; anything more is softmax over the classes
    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        model.add(layers.Dense(num_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=ann_cfg["learning_rate"]),
        loss=loss,
        metrics=metrics,
    )
    return model


class ANNModel:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.model = None
        self.history = None

    def build(self, input_dim, num_classes):
        self.model = build_ann(input_dim, num_classes, self.config)
        logger.info("ANN built: input_dim=%d, num_classes=%d", input_dim, num_classes)

    def train(self, X_train, y_train, X_val, y_val):
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        ann_cfg = self.config["ann"]
        # stop early if val loss stalls, and back off the LR when it plateaus
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=ann_cfg["early_stopping_patience"],
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=ann_cfg["epochs"],
            batch_size=ann_cfg["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

    def predict(self, X):
        raw = self.model.predict(X)
        # single-output head means binary -> threshold at 0.5, otherwise argmax
        if raw.shape[1] == 1:
            return (raw.squeeze() > 0.5).astype(int)
        return np.argmax(raw, axis=1)

    def predict_proba(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, class_names):
        from sklearn.metrics import classification_report, confusion_matrix

        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\n%s", classification_report(y_test, y_pred, target_names=class_names))
        return {"classification_report": report, "confusion_matrix": cm.tolist()}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info("saved ANN model to %s", path)

    def load(self, path):
        from tensorflow import keras
        self.model = keras.models.load_model(path)
        logger.info("loaded ANN model from %s", path)
