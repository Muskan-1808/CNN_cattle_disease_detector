
import os
import mimetypes
import requests
import streamlit as st

# ==================================================
# 1. CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CattleCare AI | Prediction",
    page_icon="🐄",
    layout="wide"
)

API_URL = os.getenv(
    "CATTLE_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

MAX_FILE_SIZE = 10 * 1024 * 1024

CLASS_LABELS = {
    "lumpy_skin_disease": "Lumpy Skin Disease",
    "foot_and_mouth_disease": "Foot and Mouth Disease",
    "healthy_cow": "Healthy Cow"
}


def readable_class(class_name):
    return CLASS_LABELS.get(
        class_name,
        class_name.replace("_", " ").title()
    )


# ==================================================
# 2. LOGIN CHECK
# ==================================================

if not st.session_state.get("logged_in", False):
    st.warning("Please log in to access this page.")
    st.info("Please return to the Login page to continue.")
    st.stop()


# ==================================================
# 3. CSS STYLING
# ==================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef8ef, #f8fcf8);
    font-family: Arial, sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(120deg, #123d2a, #25834c);
    padding: 30px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
}

.hero p {
    color: #e2f5e7;
}

.panel {
    background: white;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #d9eadc;
    box-shadow: 0 8px 25px rgba(20, 80, 40, 0.08);
    margin-bottom: 20px;
}

.result {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border-left: 6px solid #25834c;
    box-shadow: 0 8px 25px rgba(20, 80, 40, 0.08);
}

.stButton > button {
    background: #21854b;
    color: white;
    border-radius: 10px;
    border: none;
    min-height: 45px;
    font-weight: bold;
}

.stButton > button:hover {
    background: #176b39;
    color: white;
    border: none;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. HEADER
# ==================================================

user_name = st.session_state.get("user_name", "User")

st.markdown("""
<div class="hero">
    <h1>🐄 Cattle Disease Detector</h1>
    <p>Smart cattle disease detection powered by CNN</p>
</div>
""", unsafe_allow_html=True)

st.write(f"Welcome, {user_name}!")


# ==================================================
# 5. TOP BAR AND LOGOUT
# ==================================================

left, right = st.columns([4, 1])

with left:
    st.markdown("## 🌿 Disease Prediction")

with right:

    if st.button("Logout"):

        st.session_state["logged_in"] = False
        st.session_state["user_email"] = None
        st.session_state["user_name"] = None

        st.success("You have been logged out successfully.")
        st.stop()


# ==================================================
# 6. API AND MYSQL CONNECTION TEST
# ==================================================

api_connected = False

try:

    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    health_response.raise_for_status()

    health_data = health_response.json()

    if (
        health_data.get("status") == "running"
        and health_data.get("database") == "connected"
        and health_data.get("model_loaded")
    ):

        st.success("CNN API and MySQL are connected.")
        api_connected = True

    else:

        st.warning(
            "API is running, but the CNN model or database "
            "needs attention."
        )

except requests.RequestException:

    st.error(
        "Cannot connect to FastAPI. "
        "Please start your cattle FastAPI backend first."
    )


# ==================================================
# 7. IMAGE UPLOAD
# ==================================================

# Removed gap="large" for Streamlit 1.10 compatibility.

upload_col, result_col = st.columns([1, 1])

with upload_col:

    st.markdown("""
    <div class="panel">
        <h3 style="color:#174b32;">📷 Upload Cattle Image</h3>
        <p>Choose a clear JPG, JPEG, or PNG image.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="cattle_image"
    )

    image_bytes = None

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        if len(image_bytes) > MAX_FILE_SIZE:

            st.error("Image size must be below 10 MB.")
            image_bytes = None

        else:

            st.image(
                image_bytes,
                caption="Original uploaded cattle image",
                use_column_width=True
            )

    predict_button = st.button("🔍 Analyze Image")


# ==================================================
# 8. CNN PREDICTION THROUGH FASTAPI
# ==================================================

with result_col:

    st.markdown("""
    <div class="panel">
        <h3 style="color:#174b32;">📊 Prediction Results</h3>
        <p>Results are generated by your trained CNN model.</p>
    </div>
    """, unsafe_allow_html=True)

    if predict_button:

        if uploaded_file is None:

            st.warning("Please upload an image first.")

        elif image_bytes is None:

            st.error("Invalid image or image exceeds 10 MB.")

        elif not api_connected:

            st.error(
                "FastAPI, model, or database is not ready. "
                "Please check your backend."
            )

        else:

            try:

                with st.spinner(
                    "Sending image to FastAPI and analyzing with CNN..."
                ):

                    content_type = (
                        uploaded_file.type
                        or mimetypes.guess_type(
                            uploaded_file.name
                        )[0]
                        or "application/octet-stream"
                    )

                    response = requests.post(
                        f"{API_URL}/predict",
                        files={
                            "file": (
                                uploaded_file.name,
                                image_bytes,
                                content_type
                            )
                        },
                        timeout=120
                    )

                    response.raise_for_status()

                    result = response.json()

                # ----------------------------------
                # PREDICTION RESULT
                # ----------------------------------

                predicted_class = result["predicted_class"]

                readable = readable_class(predicted_class)

                confidence = float(result["confidence"])

                st.markdown(f"""
                <div class="result">
                    <p style="color:#55705e;">
                        PREDICTED CLASS
                    </p>

                    <h2 style="color:#174b32;">
                        {readable}
                    </h2>

                    <h3 style="color:#21854b;">
                        Model score: {confidence:.2f}%
                    </h3>
                </div>
                """, unsafe_allow_html=True)

                st.success(
                    f"Prediction ID: {result['prediction_id']}"
                )

                # ----------------------------------
                # UNCERTAINTY WARNING
                # ----------------------------------

                if confidence < 60:

                    st.warning(
                        "The model is uncertain. Try a clearer image "
                        "and consult a veterinarian if disease is suspected."
                    )

                # ----------------------------------
                # CLASS-WISE SCORES
                # ----------------------------------

                st.markdown("### Class-wise scores")

                class_scores = result["all_class_scores"]

                for class_name, score in class_scores.items():

                    label = readable_class(class_name)

                    score = float(score)

                    st.write(
                        f"**{label}: {score:.2f}%**"
                    )

                    # Integer percentage for older Streamlit versions
                    progress_value = int(
                        min(max(score, 0.0), 100.0)
                    )

                    st.progress(progress_value)

                # ----------------------------------
                # ORIGINAL IMAGE
                # ----------------------------------

                st.markdown("### Saved Original Image")

                st.image(
                    image_bytes,
                    caption="Original image preserved by FastAPI",
                    use_column_width=True
                )

                st.caption(
                    "Original image is stored in the backend uploads folder."
                )

                st.caption(
                    "Scores are model outputs, not confirmed diagnoses "
                    "or necessarily calibrated probabilities."
                )

                st.info(
                    "AI estimate only. Consult a qualified veterinarian "
                    "for suspected cattle disease."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "Prediction request timed out. Please try again."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "FastAPI is not running. Start your backend first."
                )

            except requests.exceptions.HTTPError as exc:

                try:
                    detail = response.json().get(
                        "detail",
                        str(exc)
                    )
                except ValueError:
                    detail = str(exc)

                st.error(f"API Error: {detail}")

            except (KeyError, ValueError, TypeError) as exc:

                st.error(
                    f"Invalid response received from FastAPI: {exc}"
                )

            except requests.RequestException as exc:

                st.error(f"Request failed: {exc}")

    else:

        st.info(
            "Upload a cattle image and click Analyze Image "
            "to see the CNN prediction."
        )


# ==================================================
# 9. PREDICTION HISTORY FROM MYSQL
# ==================================================

st.markdown("---")

st.markdown("## 📚 Prediction History")

if st.button("Refresh Prediction History"):

    try:

        with st.spinner("Retrieving history from MySQL..."):

            history_response = requests.get(
                f"{API_URL}/history",
                timeout=20
            )

            history_response.raise_for_status()

            history = history_response.json()

        if not history:

            st.info("No prediction history available yet.")

        else:

            st.success(
                f"Total predictions: {len(history)}"
            )

            for record in history:

                with st.expander(
                    f"Prediction #{record['id']} - "
                    f"{readable_class(record['predicted_class'])}"
                ):

                    history_col1, history_col2 = st.columns(2)

                    with history_col1:

                        st.write(
                            f"**Filename:** {record['filename']}"
                        )

                        st.write(
                            f"**Predicted class:** "
                            f"{readable_class(record['predicted_class'])}"
                        )

                        st.write(
                            f"**Confidence:** "
                            f"{float(record['confidence']):.2f}%"
                        )

                        created_at = record.get("created_at")

                        if created_at:
                            st.write(f"**Date:** {created_at}")

                    with history_col2:

                        image_url = record.get("image_url")

                        if image_url:

                            st.image(
                                image_url,
                                caption="Saved original image",
                                use_column_width=True
                            )

                        else:

                            st.caption("Image URL not available.")

    except requests.RequestException as exc:

        st.error(
            f"Could not retrieve MySQL history: {exc}"
        )

    except (KeyError, ValueError, TypeError) as exc:

        st.error(
            f"Invalid history response: {exc}"
        )


# ==================================================
# 10. FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "© 2026 CattleCare AI | CNN Image Classification Project"
)

st.caption(
    "Educational prototype only. Consult a veterinarian "
    "for suspected cattle disease."
)
