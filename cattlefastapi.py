
import json
import io
import os
import uuid

import numpy as np
import tensorflow as tf

from pathlib import Path
from datetime import datetime, timezone

from PIL import Image, ImageOps, UnidentifiedImageError

from fastapi import FastAPI, File, HTTPException, UploadFile, Request

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    text
)

from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError


# ==================================================
# 1. PATHS AND CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "cattle_disease_cnn.keras"
LABEL_PATH = BASE_DIR / "class_names.json"

IMAGE_SIZE = (128, 128)
MAX_FILE_SIZE = 10 * 1024 * 1024

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. MYSQL CONNECTION
# ==================================================

DB_PASSWORD = os.getenv("DB_PASSWORD")

if not DB_PASSWORD:
    raise RuntimeError(
        "DB_PASSWORD environment variable is missing."
    )

DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username="root",
    password=DB_PASSWORD,
    host="localhost",
    port=3306,
    database="cattle_disease_db"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


# ==================================================
# 3. PREDICTION TABLE
# ==================================================

class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    filename = Column(String(255), nullable=False)

    image_path = Column(String(500), nullable=False)

    predicted_class = Column(String(100), nullable=False)

    confidence = Column(Float, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )


# ==================================================
# 4. CREATE MYSQL TABLE
# ==================================================

try:

    Base.metadata.create_all(bind=engine)

    print("MySQL connected successfully!")
    print("Predictions table ready!")

except SQLAlchemyError as exc:

    raise RuntimeError(
        f"MySQL connection failed: {exc}"
    ) from exc


# ==================================================
# 5. LOAD CLASS NAMES
# ==================================================

if not LABEL_PATH.exists():
    raise FileNotFoundError(LABEL_PATH)

with open(LABEL_PATH, "r", encoding="utf-8-sig") as file:
    class_names = json.load(file)

if not isinstance(class_names, list) or len(class_names) != 3:
    raise ValueError("Expected exactly three classes.")

if not all(
    isinstance(name, str) and name.strip()
    for name in class_names
):
    raise ValueError("Invalid class names.")


# ==================================================
# 6. LOAD CNN MODEL
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        "CNN output count does not match class names."
    )

print("CNN model loaded successfully!")
print("Class order:", class_names)


# ==================================================
# 7. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Cattle Disease Detection API",
    description="CNN cattle disease detection with MySQL",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads"
)


# ==================================================
# 8. HOME ENDPOINT
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Cattle Disease Detection API is running",
        "model": MODEL_PATH.name,
        "classes": class_names,
        "database": "cattle_disease_db",
        "image_size": IMAGE_SIZE,
        "docs": "/docs"
    }


# ==================================================
# 9. HEALTH CHECK
# ==================================================

@app.get("/health")
def health():

    try:

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        database_status = "connected"

    except SQLAlchemyError:

        database_status = "disconnected"

    return {
        "status": "running",
        "model_loaded": model is not None,
        "database": database_status,
        "classes": class_names
    }


# ==================================================
# 10. IMAGE PREPROCESSING
# ==================================================

def preprocess_image(image_bytes):

    try:

        with Image.open(io.BytesIO(image_bytes)) as image:

            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

            # Assumes model contains Rescaling(1./255).

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            return image_array

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as exc:

        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        ) from exc


# ==================================================
# 11. PREDICTION ENDPOINT
# ==================================================

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):

    image_path = None

    try:

        allowed_types = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp"
        }

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Upload JPG, PNG or WEBP image."
            )

        image_bytes = await file.read(MAX_FILE_SIZE + 1)

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="Image size must be below 10 MB."
            )

        # Preprocess image for CNN
        image_array = preprocess_image(image_bytes)

        # CNN prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )

        predictions = np.asarray(predictions)

        if predictions.shape != (1, len(class_names)):
            raise ValueError("Unexpected prediction output shape.")

        scores = predictions[0]

        if not np.all(np.isfinite(scores)):
            raise ValueError("Invalid CNN prediction scores.")

        # Predicted class
        predicted_index = int(np.argmax(scores))
        predicted_class = class_names[predicted_index]

        confidence = float(scores[predicted_index]) * 100

        # Save original uploaded image
        image_name = (
            f"{uuid.uuid4()}{allowed_types[file.content_type]}"
        )

        image_path = UPLOAD_DIR / image_name
        image_path.write_bytes(image_bytes)

        image_url = (
            str(request.base_url).rstrip("/")
            + f"/uploads/{image_name}"
        )

        # Save prediction to MySQL
        db = SessionLocal()

        try:

            record = Prediction(
                filename=Path(
                    file.filename or "uploaded_image"
                ).name,
                image_path=str(image_path),
                predicted_class=predicted_class,
                confidence=confidence
            )

            db.add(record)
            db.commit()
            db.refresh(record)

            prediction_id = record.id
            created_at = record.created_at

        except SQLAlchemyError as exc:

            db.rollback()
            image_path.unlink(missing_ok=True)
            image_path = None

            raise HTTPException(
                status_code=500,
                detail="Failed to save prediction in MySQL."
            ) from exc

        finally:
            db.close()

        return {
            "message": "Prediction completed successfully",
            "prediction_id": prediction_id,
            "filename": Path(
                file.filename or "uploaded_image"
            ).name,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "all_class_scores": {
                class_names[i]: round(
                    float(scores[i]) * 100, 2
                )
                for i in range(len(class_names))
            },
            "image_url": image_url,
            "original_image_preserved": True,
            "cnn_input_size": "128x128",
            "created_at": created_at,
            "note": "AI estimate, not a veterinary diagnosis."
        }

    except HTTPException:
        raise

    except Exception as exc:

        print("Prediction error:", exc)

        if image_path is not None:
            image_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check FastAPI logs."
        ) from exc

    finally:
        await file.close()


# ==================================================
# 12. PREDICTION HISTORY
# ==================================================

@app.get("/history")
def get_history(request: Request):

    db = SessionLocal()

    try:

        records = db.query(
            Prediction
        ).order_by(
            Prediction.id.desc()
        ).all()

        history = []

        for record in records:

            filename = Path(record.image_path).name

            image_url = (
                str(request.base_url).rstrip("/")
                + f"/uploads/{filename}"
            )

            history.append({
                "id": record.id,
                "filename": record.filename,
                "predicted_class": record.predicted_class,
                "confidence": round(record.confidence, 2),
                "image_url": image_url,
                "created_at": record.created_at
            })

        return history

    except SQLAlchemyError as exc:

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve prediction history."
        ) from exc

    finally:
        db.close()


# ==================================================
# 13. DATABASE TEST
# ==================================================

@app.get("/database")
def database_test():

    try:

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "database": "MySQL",
            "status": "Connected successfully",
            "name": "cattle_disease_db"
        }

    except SQLAlchemyError as exc:

        raise HTTPException(
            status_code=500,
            detail="MySQL connection failed."
        ) from exc
