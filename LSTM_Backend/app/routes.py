# app/routes.py

from flask import Blueprint, request, jsonify
from app.services.feature_service import prepare_features
from app.services.model_service import get_model_engine
from app.utils.database import get_recent_history, add_new_record

import logging
import gc

main_bp = Blueprint('main', __name__)


@main_bp.route('/predict', methods=['POST'])
def predict_energy():
    try:
        data = request.get_json()

        # 🔹 Validate input
        company_name = data.get('company_name')
        if not company_name:
            return jsonify({"error": "Missing company_name"}), 400

        # 🔹 Fetch history (needed for LSTM window)
        history_df = get_recent_history(company_name, limit=48)

        if len(history_df) < 24:
            add_new_record(data)
            return jsonify({
                "status": "pending",
                "message": f"Insufficient history. Found {len(history_df)}/24 steps."
            }), 200

        # 🔹 Feature engineering
        feature_matrix = prepare_features(history_df, data['max_capacity'])

        # 🔥 Load model only when needed
        engine = get_model_engine()
        prediction_kw = engine.predict(feature_matrix)

        # 🔥 CRITICAL: Free TensorFlow memory properly
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass

        # 🔥 Delete instance + force garbage collection
        del engine
        gc.collect()

        # 🔹 Save prediction
        add_new_record(data, prediction=prediction_kw)

        return jsonify({
            "status": "success",
            "company_name": company_name,
            "prediction": prediction_kw,
            "unit": "kW"
        }), 200

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        return jsonify({
            "error": "Internal Server Error",
            "details": str(e)
        }), 500


@main_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "service": "lstm-backend",
        "note": "Model loads lazily per request"
    }), 200