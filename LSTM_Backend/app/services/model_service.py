# app/services/model_service.py

class ModelService:
    def __init__(self):
        # Do NOT load anything heavy here
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None

    def load(self):
        # Lazy import + lazy load
        if self.model is None:
            import tensorflow as tf
            import joblib
            import os

            print("Loading LSTM model...")

            model_path = os.path.join('models', 'energy_lstm_model.h5')
            f_scaler_path = os.path.join('models', 'feature_scaler.pkl')
            t_scaler_path = os.path.join('models', 'target_scaler.pkl')

            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.feature_scaler = joblib.load(f_scaler_path)
            self.target_scaler = joblib.load(t_scaler_path)

            print("Model and scalers loaded.")

    def predict(self, feature_matrix):
        import numpy as np

        # Load only when needed
        self.load()

        # Normalize features
        scaled = self.feature_scaler.transform(feature_matrix)

        # Reshape for LSTM → (1, 24, 24)
        input_tensor = np.expand_dims(scaled, axis=0)

        # Run inference
        pred_scaled = self.model.predict(input_tensor, verbose=0)

        # Reverse scaling
        result = self.target_scaler.inverse_transform(pred_scaled)[0][0]

        return round(float(result), 3)


def get_model_engine():
    # Always create fresh instance (no global memory retention)
    return ModelService()