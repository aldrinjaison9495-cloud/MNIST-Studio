
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import textwrap

from PIL import Image
from streamlit_drawable_canvas import st_canvas


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MNIST Studio",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. HTML RENDERING FUNCTION
# =====================================================

def render_html(content):
    st.html(textwrap.dedent(content).strip())


# =====================================================
# 3. PREMIUM DARK CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: #090E1B;
    color: #F8FAFC;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #10182A;
    border-right: 1px solid #25314A;
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

p {
    color: #AAB6CC;
}

[data-testid="stMetric"] {
    background: #151F34;
    border: 1px solid #293954;
    padding: 20px;
    border-radius: 15px;
}

[data-testid="stMetricValue"] {
    color: #A78BFA;
}

.stButton > button {
    width: 100%;
    min-height: 43px;
    border: 1px solid #5145CD;
    border-radius: 11px;
    background: linear-gradient(110deg, #6366F1, #8B5CF6);
    color: white;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(110deg, #4F46E5, #7C3AED);
    color: white;
    border-color: #A78BFA;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: #111A2D;
    padding: 8px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    color: #AAB6CC;
    border-radius: 9px;
    padding: 10px 18px;
}

.stTabs [aria-selected="true"] {
    background: #29245B;
    color: white;
}

[data-testid="stFileUploader"] {
    background: #151F34;
    border: 1px dashed #5145CD;
    padding: 15px;
    border-radius: 14px;
}

hr {
    border-color: #26344C;
}

.custom-card {
    background: linear-gradient(145deg, #17223A, #111A2D);
    border: 1px solid #293954;
    border-radius: 17px;
    padding: 22px;
    margin-bottom: 12px;
}

.result-card {
    background: linear-gradient(145deg, #24204F, #151B35);
    border: 1px solid #6657D9;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 8px 35px rgba(99,102,241,0.12);
}

.result-number {
    font-size: 100px;
    font-weight: 800;
    line-height: 1.1;
    color: #C4B5FD;
    margin: 5px 0;
}

.small-label {
    color: #A5B4FC;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
}

.muted-text {
    color: #8492AB;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. LOAD CNN MODEL
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.keras")


model = load_model()


# =====================================================
# 5. IMAGE PREPROCESSING
# =====================================================

def preprocess_digit(image):

    image = np.array(image)

    # Convert RGB/RGBA to grayscale
    if image.ndim == 3:

        if image.shape[2] == 4:
            image = image[:, :, 0]

        else:
            image = np.mean(
                image[:, :, :3],
                axis=2
            )

    image = image.astype("float32")

    # Invert white background images
    if image.mean() > 127:
        image = 255 - image

    # Find digit boundaries
    coords = np.argwhere(image > 30)

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop digit
    cropped = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    digit_img = Image.fromarray(
        np.clip(cropped, 0, 255).astype("uint8")
    )

    # Resize while maintaining aspect ratio
    digit_img.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Create 28x28 black image
    final_img = Image.new(
        "L",
        (28, 28),
        0
    )

    # Center digit
    x_offset = (28 - digit_img.width) // 2
    y_offset = (28 - digit_img.height) // 2

    final_img.paste(
        digit_img,
        (x_offset, y_offset)
    )

    # Normalize pixel values
    result = (
        np.array(final_img).astype("float32") / 255.0
    )

    return result.reshape(1, 28, 28, 1)


# =====================================================
# 6. PREDICTION FUNCTION
# =====================================================

def predict_digit(processed_image):

    prediction = model.predict(
        processed_image,
        verbose=0
    )[0]

    digit = int(np.argmax(prediction))

    confidence = float(
        np.max(prediction) * 100
    )

    return digit, confidence, prediction


# =====================================================
# 7. SESSION STATE
# =====================================================

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =====================================================
# 8. SIDEBAR
# =====================================================

with st.sidebar:

    render_html("""
        <h2 style="color:#A78BFA;">🔢 MNIST Studio</h2>

        <p style="font-size:11px;letter-spacing:2px;">
            INTELLIGENT DIGIT RECOGNITION
        </p>

        <hr>

        <h3>🧠 Model Information</h3>

        <div class="custom-card">
            <p class="small-label">MODEL STATUS</p>
            <h3 style="color:#34D399;">● Active</h3>
            <p class="muted-text">CNN model loaded successfully</p>
        </div>
    """)

    st.write("**Algorithm:** CNN")
    st.write("**Dataset:** MNIST")
    st.write("**Input Size:** 28 × 28")
    st.write("**Classes:** 0–9")
    st.write("**Test Accuracy:** 98.72%")

    st.divider()

    st.markdown("### ⚙️ Technologies")

    st.write("🐍 Python")
    st.write("🧠 TensorFlow")
    st.write("🔢 NumPy")
    st.write("📊 Pandas")
    st.write("⚡ Streamlit")

    st.divider()

    st.caption(
        "Deep Learning | Handwritten Digit Recognition"
    )


# =====================================================
# 9. MAIN HEADER
# =====================================================

render_html("""
    <div style="margin-bottom:25px;">

        <p style="
            color:#A78BFA;
            font-size:12px;
            letter-spacing:3px;
            font-weight:700;
        ">
            DEEP LEARNING APPLICATION
        </p>

        <h1 style="
            font-size:42px;
            margin-bottom:5px;
            background:linear-gradient(
                90deg,
                #FFFFFF,
                #A78BFA,
                #60A5FA
            );
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
        ">
            MNIST  Studio
        </h1>

        <p style="font-size:15px;color:#94A3B8;">
            Transform handwritten digits into intelligent predictions.
            Powered by Convolutional Neural Networks.
        </p>

    </div>
""")


# =====================================================
# 10. DASHBOARD METRICS
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Accuracy", "98.72%", "MNIST Test Dataset")

with col2:
    st.metric("Digit Classes", "10", "Digits 0 to 9")

with col3:
    st.metric("Input Resolution", "28 × 28", "Grayscale")

st.divider()


# =====================================================
# 11. APPLICATION TABS
# =====================================================

draw_tab, upload_tab, history_tab, about_tab = st.tabs([
    "✏️ Drawing Studio",
    "📁 Image Upload",
    "📊 History",
    "ℹ️ About Model"
])


# =====================================================
# 12. DRAWING STUDIO
# =====================================================

with draw_tab:

    left, right = st.columns([1.05, 1], gap="large")

    with left:

        st.markdown("### ✏️ Drawing Workspace")

        st.caption(
            "Draw a single handwritten digit between 0 and 9."
        )

        brush_size = st.slider(
            "Brush Thickness",
            min_value=5,
            max_value=30,
            value=15
        )

        canvas = st_canvas(
            fill_color="rgba(255,255,255,0)",
            stroke_width=brush_size,
            stroke_color="#FFFFFF",
            background_color="#080D18",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            return_image_data=True
        )

        st.write("")

        btn1, btn2 = st.columns(2)

        with btn1:

            predict_button = st.button(
                "✦ Predict Digit",
                key="draw_predict"
            )

        with btn2:

            clear_button = st.button(
                "↻ Clear Canvas",
                key="clear_canvas"
            )

        if clear_button:

            st.session_state.canvas_key += 1
            st.session_state.last_result = None
            st.rerun()

    with right:

        st.markdown("### 🧠  Prediction")

        if predict_button:

            if canvas.image_data is None:

                st.warning("Please draw a digit first.")

            else:

                raw_image = canvas.image_data[:, :, 0]

                processed_image = preprocess_digit(raw_image)

                if processed_image is None:

                    st.warning("No digit detected.")

                else:

                    with st.spinner("CNN is analyzing your digit..."):

                        digit, confidence, probabilities = (
                            predict_digit(processed_image)
                        )

                    st.session_state.last_result = {
                        "digit": digit,
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "processed_image": processed_image
                    }

                    st.session_state.history.append({
                        "Source": "Drawing",
                        "Digit": digit,
                        "Confidence": round(confidence, 2)
                    })

        result = st.session_state.last_result

        if result is not None:

            digit = result["digit"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]
            processed_image = result["processed_image"]

            render_html(f"""
                <div class="result-card">
                    <p class="small-label">PREDICTED DIGIT</p>
                    <div class="result-number">{digit}</div>
                    <p style="color:#A5B4FC;">
                        AI Prediction Completed
                    </p>
                </div>
            """)

            st.write("")

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(min(confidence / 100, 1.0))

            st.markdown("#### Processed MNIST Image")

            st.image(
                processed_image[0, :, :, 0],
                width=120,
                clamp=True
            )

            st.markdown("#### Top 3 Predictions")

            top_indices = np.argsort(probabilities)[::-1][:3]

            for index in top_indices:

                probability = float(probabilities[index] * 100)

                st.write(f"**Digit {index}**")

                st.progress(min(probability / 100, 1.0))

                st.caption(f"{probability:.2f}% confidence")

            st.markdown("#### Probability Distribution")

            chart_data = pd.DataFrame({
                "Digit": [str(i) for i in range(10)],
                "Probability": probabilities * 100
            }).set_index("Digit")

            st.bar_chart(chart_data)

        else:

            render_html("""
                <div class="custom-card"
                     style="text-align:center;padding:55px 20px;">

                    <h1 style="font-size:65px;">🧠</h1>

                    <h3>Waiting for Input</h3>

                    <p class="muted-text">
                        Draw a digit on the left and click
                        Predict Digit to see the AI result.
                    </p>

                </div>
            """)


# =====================================================
# 13. IMAGE UPLOAD
# =====================================================

with upload_tab:

    st.markdown("### 📁 Upload Handwritten Image")

    st.caption(
        "Upload an image containing a single handwritten digit."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        key="uploaded_digit"
    )

    if uploaded_file is not None:

        uploaded_image = Image.open(
            uploaded_file
        ).convert("L")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Original Image")

            st.image(
                uploaded_image,
                use_container_width=True
            )

        with col2:

            st.markdown("#### Image Information")

            st.write(f"Width: {uploaded_image.width}px")
            st.write(f"Height: {uploaded_image.height}px")
            st.write("Format: Grayscale")

        if st.button("🚀 Analyze Image", key="upload_predict"):

            processed_image = preprocess_digit(uploaded_image)

            if processed_image is None:

                st.warning("No digit detected.")

            else:

                with st.spinner("Analyzing image..."):

                    digit, confidence, probabilities = (
                        predict_digit(processed_image)
                    )

                st.success(f"Predicted Digit: {digit}")

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                st.markdown("#### Processed Image")

                st.image(
                    processed_image[0, :, :, 0],
                    width=140,
                    clamp=True
                )

                st.markdown("#### Top 3 Predictions")

                top_indices = np.argsort(probabilities)[::-1][:3]

                for index in top_indices:

                    st.write(
                        f"Digit {index}: "
                        f"{probabilities[index] * 100:.2f}%"
                    )

                st.markdown("#### Probability Distribution")

                chart_data = pd.DataFrame({
                    "Digit": [str(i) for i in range(10)],
                    "Probability": probabilities * 100
                }).set_index("Digit")

                st.bar_chart(chart_data)

                st.session_state.history.append({
                    "Source": "Image Upload",
                    "Digit": digit,
                    "Confidence": round(confidence, 2)
                })


# =====================================================
# 14. PREDICTION HISTORY
# =====================================================

with history_tab:

    st.markdown("### 📊 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Predictions", len(history_df))

        with col2:
            st.metric(
                "Average Confidence",
                f"{history_df['Confidence'].mean():.2f}%"
            )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("🗑️ Clear History"):

            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

    else:

        st.info(
            "No prediction history yet. "
            "Start drawing or uploading digits."
        )


# =====================================================
# 15. ABOUT MODEL
# =====================================================

with about_tab:

    st.markdown("### 🧠 About the CNN Model")

    st.write(
        "This application uses a trained Convolutional Neural "
        "Network to recognize handwritten digits."
    )

    architecture = pd.DataFrame({

        "Layer": [
            "Input Layer",
            "Convolution2D",
            "MaxPooling2D",
            "Flatten",
            "Dense",
            "Output Dense"
        ],

        "Description": [
            "28 × 28 grayscale image",
            "32 filters, 3 × 3 kernel",
            "2 × 2 pooling",
            "Feature vector conversion",
            "100 neurons with ReLU",
            "10 neurons with Softmax"
        ]

    })

    st.markdown("#### Model Architecture")

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Training Images", "60,000")

    with col2:
        st.metric("Testing Images", "10,000")

    st.markdown("#### Model Performance")

    st.metric("MNIST Test Accuracy", "98.72%")

    st.progress(0.9872)

    st.caption(
        "Based on your reported MNIST test evaluation."
    )


# =====================================================
# 16. FOOTER
# =====================================================

st.divider()

render_html("""
    <div style="text-align:center;padding:20px;">

        <h3 style="color:#A78BFA;">
            MNIST  Studio
        </h3>

        <p style="color:#64748B;font-size:13px;">
            Built with Python · TensorFlow · CNN · Streamlit
        </p>

        <p style="color:#475569;font-size:12px;">
            Handwritten Digit Recognition Project
        </p>

    </div>
""")
