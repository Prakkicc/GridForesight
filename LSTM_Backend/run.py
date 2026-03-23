# run.py

import os

# Reduce TensorFlow logging + memory overhead
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("[1/6] Starting ML Backend...")

from flask import Flask
from flask_cors import CORS

print("[2/6] Loading routes...")
from app.routes import main_bp

print("[3/6] Loading database...")
from app.utils.database import Base, engine


def create_app():
    app = Flask(__name__)
    CORS(app)

    with app.app_context():
        print("[4/6] Verifying database...")
        Base.metadata.create_all(bind=engine)
        print("[5/6] Database ready")

    app.register_blueprint(main_bp)
    return app


# For gunicorn
app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"[6/6] Server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)