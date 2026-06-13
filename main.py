import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gdown

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="🌿 AI Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    background: linear-gradient(135deg,#0f9d58,#34a853);
    padding: 35px;
    border-radius: 18px;
    text-align:center;
    color:white;
    margin-bottom:20px;
}

.prediction-card {
    background:#eef8ef;
    padding:20px;
    border-radius:15px;
    border-left:6px solid #0f9d58;
}

.footer {
    text-align:center;
    color:gray;
    margin-top:50px;
}

.sidebar-title {
    color:#0f9d58;
    font-size:24px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

working_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    working_dir,
    "plant_model.h5"
)

class_indices_path = os.path.join(
    working_dir,
    "class_indices.json"
)

assets_dir = os.path.join(
    working_dir,
    "assets"
)

examples_path = os.path.join(
    assets_dir,
    "class_examples"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():

    if not os.path.exists(model_path):

        url = "https://drive.google.com/uc?id=1Ww6IHkPBzK7jJsMsQMWUhax39y3TVnA1"

        with st.spinner("Downloading model..."):

            gdown.download(
                url,
                model_path,
                quiet=False
            )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model

model = load_model()

# ---------------------------------------------------
# LOAD CLASS LABELS
# ---------------------------------------------------

with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# ---------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------

def load_and_preprocess_image(
        image,
        target_size=(224, 224)
):

    img = Image.open(image).convert("RGB")
    img = img.resize(target_size)

    img_array = np.array(img).astype("float32")
    img_array /= 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array

# ---------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------

def predict_image_class(
        model,
        image,
        class_indices
):

    processed_img = load_and_preprocess_image(
        image
    )

    predictions = model.predict(
        processed_img,
        verbose=0
    )

    top_indices = predictions[0].argsort()[-3:][::-1]

    top_classes = [
        class_indices[str(i)]
        for i in top_indices
    ]

    top_confidences = [
        predictions[0][i] * 100
        for i in top_indices
    ]

    predicted_class = top_classes[0]
    confidence = top_confidences[0]

    return (
        predicted_class,
        confidence,
        top_classes,
        top_confidences
    )


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.markdown(
    "<div class='sidebar-title'>🌿 Plant AI</div>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔬 Disease Detection",
        "📊 Dashboard",
        "📚 Disease Gallery",
        "👨‍💻 About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success(
    "✅ Model Loaded"
)

st.sidebar.info(
    "MobileNetV2 • TensorFlow"
)

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero">
            <h1>🌿 AI Plant Disease Detection System</h1>
            <h3>Deep Learning Powered Crop Health Monitoring</h3>
            <p>
            Upload a leaf image and instantly detect plant diseases
            using TensorFlow and MobileNetV2.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    bg_path = os.path.join(
        assets_dir,
        "leaf_bg.jpg"
    )

    if os.path.exists(bg_path):

        st.image(
            bg_path,
            use_container_width=True
        )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Disease Classes",
            "38"
        )

    with col2:
        st.metric(
            "Architecture",
            "MobileNetV2"
        )

    with col3:
        st.metric(
            "Input Size",
            "224 × 224"
        )

    with col4:
        st.metric(
            "Deployment",
            "Cloud"
        )

    st.divider()

    st.header("🌟 Features")

    feature1, feature2, feature3 = st.columns(3)

    with feature1:

        upload_icon = os.path.join(
            assets_dir,
            "upload_icon.png"
        )

        if os.path.exists(upload_icon):

            st.image(
                upload_icon,
                width=80
            )

        st.subheader(
            "Multi Image Upload"
        )

        st.write(
            "Upload one or multiple leaf images "
            "for instant disease classification."
        )

    with feature2:

        chart_icon = os.path.join(
            assets_dir,
            "chart_icon.png"
        )

        if os.path.exists(chart_icon):

            st.image(
                chart_icon,
                width=80
            )

        st.subheader(
            "Visual Analytics"
        )

        st.write(
            "Interactive charts show prediction "
            "confidence and disease probabilities."
        )

    with feature3:

        class_icon = os.path.join(
            assets_dir,
            "class_icon.png"
        )

        if os.path.exists(class_icon):

            st.image(
                class_icon,
                width=80
            )

        st.subheader(
            "Disease Gallery"
        )

        st.write(
            "Browse disease classes and example "
            "leaf images."
        )

    st.divider()

    st.subheader(
        "🌱 Why This Project?"
    )

    st.info(
        """
        Plant diseases significantly affect crop yield.
        This AI system helps farmers, students and
        researchers identify diseases quickly using
        deep learning and computer vision.
        """
    )

    st.success(
        "Ready to test the model? "
        "Open the Disease Detection page."
    )

# ---------------------------------------------------
# DISEASE DETECTION PAGE
# ---------------------------------------------------

elif page == "🔬 Disease Detection":

    st.title("🔬 Plant Disease Detection")

    st.markdown(
        """
        Upload one or more leaf images and let the AI
        identify diseases instantly.
        """
    )

    uploaded_images = st.file_uploader(
        "📤 Upload Leaf Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_images:

        results = []

        for uploaded_image in uploaded_images:

            st.divider()

            st.subheader(
                f"📷 {uploaded_image.name}"
            )

            col1, col2 = st.columns([1, 2])

            # ---------------------------
            # IMAGE PREVIEW
            # ---------------------------

            with col1:

                st.image(
                    uploaded_image,
                    caption="Uploaded Leaf",
                    use_container_width=True
                )

            # ---------------------------
            # PREDICTION
            # ---------------------------

            with col2:

                with st.spinner(
                    "🧠 Analyzing Leaf..."
                ):

                    (
                        predicted_class,
                        confidence,
                        top_classes,
                        top_confidences
                    ) = predict_image_class(
                        model,
                        uploaded_image,
                        class_indices
                    )

                st.markdown(
                    f"""
                    <div class="prediction-card">
                    <h3>🌿 Prediction Result</h3>
                    <p><b>Class:</b> {predicted_class}</p>
                    <p><b>Confidence:</b> {confidence:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                if "healthy" in predicted_class.lower():

                    st.success(
                        "✅ Plant appears healthy."
                    )

                else:

                    st.error(
                        "⚠️ Disease detected."
                    )

                st.write(
                    f"Confidence Score: {confidence:.2f}%"
                )

                st.progress(
                    min(
                        int(confidence),
                        100
                    )
                )

            # ---------------------------
            # TOP 3 TABLE
            # ---------------------------

            st.subheader(
                "📊 Top Predictions"
            )

            df = pd.DataFrame({

                "Class":
                    top_classes,

                "Confidence (%)":
                    [
                        round(x, 2)
                        for x in top_confidences
                    ]
            })

            st.dataframe(
                df,
                use_container_width=True
            )

            # ---------------------------
            # BAR CHART
            # ---------------------------

            fig_bar = px.bar(

                df,

                x="Class",

                y="Confidence (%)",

                color="Confidence (%)",

                text_auto=".2f",

                title="Top 3 Disease Predictions"
            )

            fig_bar.update_layout(
                height=500
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )

            # ---------------------------
            # PIE CHART
            # ---------------------------

            fig_pie = go.Figure(

                data=[
                    go.Pie(
                        labels=top_classes,
                        values=top_confidences,
                        hole=0.45
                    )
                ]
            )

            fig_pie.update_layout(
                title="Prediction Distribution",
                height=500
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )

            # ---------------------------
            # SAVE RESULTS
            # ---------------------------

            results.append({

                "Image":
                    uploaded_image.name,

                "Prediction":
                    predicted_class,

                "Confidence":
                    round(confidence, 2)
            })

        # ---------------------------
        # DOWNLOAD CSV
        # ---------------------------

        st.divider()

        st.subheader(
            "📥 Export Results"
        )

        csv_data = pd.DataFrame(
            results
        ).to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(

            label="📄 Download CSV Report",

            data=csv_data,

            file_name="plant_disease_predictions.csv",

            mime="text/csv"
        )

    else:

        st.info(
            "Upload one or more leaf images to begin."
        )


# ---------------------------------------------------
# DASHBOARD PAGE
# ---------------------------------------------------

elif page == "📊 Dashboard":

    st.title("📊 Model Dashboard")

    st.markdown(
        """
        Overview of the deployed AI model and project statistics.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Classes",
            "38"
        )

    with col2:
        st.metric(
            "Architecture",
            "MobileNetV2"
        )

    with col3:
        st.metric(
            "Input Size",
            "224×224"
        )

    with col4:
        st.metric(
            "Deployment",
            "Live"
        )

    st.divider()

    st.subheader("🧠 Model Information")

    model_info = pd.DataFrame({

        "Parameter": [
            "Framework",
            "Architecture",
            "Dataset",
            "Input Shape",
            "Classes",
            "Deployment"
        ],

        "Value": [
            "TensorFlow",
            "MobileNetV2",
            "PlantVillage",
            "224x224x3",
            "38",
            "Streamlit Cloud"
        ]

    })

    st.dataframe(
        model_info,
        use_container_width=True
    )

    st.divider()

    st.subheader("🚀 System Features")

    st.success(
        """
        ✅ Multi-Class Classification

        ✅ Deep Learning Powered

        ✅ Cloud Deployment

        ✅ CSV Export

        ✅ Interactive Charts

        ✅ Disease Gallery

        ✅ Multi Image Upload
        """
    )

    st.divider()

    st.subheader("📈 Dataset Overview")

    dataset_df = pd.DataFrame({

        "Category": [
            "Healthy Classes",
            "Diseased Classes"
        ],

        "Count": [
            12,
            26
        ]
    })

    fig_dataset = px.pie(

        dataset_df,

        names="Category",

        values="Count",

        title="Class Distribution"

    )

    st.plotly_chart(
        fig_dataset,
        use_container_width=True
    )

# ---------------------------------------------------
# DISEASE GALLERY PAGE
# ---------------------------------------------------

elif page == "📚 Disease Gallery":

    st.title("📚 Disease Gallery")

    st.markdown(
        """
        Browse disease classes and view
        example leaf images.
        """
    )

    if not os.path.exists(examples_path):

        st.warning(
            "No disease gallery found."
        )

    else:

        search = st.text_input(
            "🔍 Search Disease Class"
        )

        class_folders = sorted(
            os.listdir(examples_path)
        )

        for class_name in class_folders:

            if search:

                if search.lower() not in class_name.lower():
                    continue

            class_folder = os.path.join(
                examples_path,
                class_name
            )

            if not os.path.isdir(
                class_folder
            ):
                continue

            st.markdown(
                f"## 🌿 {class_name}"
            )

            images = [

                os.path.join(
                    class_folder,
                    image
                )

                for image in os.listdir(
                    class_folder
                )

                if image.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png"
                    )
                )
            ]

            if len(images) == 0:

                st.warning(
                    f"No images found for {class_name}"
                )

                continue

            cols = st.columns(
                min(4, len(images))
            )

            for i, image_path in enumerate(
                images[:4]
            ):

                with cols[i]:

                    st.image(
                        image_path,
                        use_container_width=True
                    )

            st.divider()

# ---------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------

elif page == "👨‍💻 About":

    st.title("👨‍💻 About Developer")

    st.markdown("""
    ## 🌿 AI Plant Disease Detection System

    This project utilizes Deep Learning and Computer Vision
    to identify plant diseases from leaf images with high accuracy.

    The objective of this system is to provide a practical,
    accessible and cloud-based solution for farmers,
    researchers and students.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("👤 Developer")

        st.info("""
        **Kisalay Shukla**

        B.Tech (Computer Science)

        Research Interests:

        • Deep Learning

        • Computer Vision

        • Plant Disease Detection

        • Transformers

        • Explainable AI
        """)

    with col2:

        st.subheader("🛠 Technologies Used")

        st.success("""
        • TensorFlow

        • Keras

        • Streamlit

        • Plotly

        • NumPy

        • Pandas

        • Pillow

        • Python
        """)

    st.divider()

    st.subheader("🔗 Project Links")

    st.markdown("""
    ### GitHub Repository

    https://github.com/kisalay12345/pdd

    ### Live Application

    https://findleafdisease.streamlit.app/
    """)

    st.divider()

    st.subheader("📊 Project Highlights")

    st.write("""
    • 38 Disease Categories

    • MobileNetV2 Architecture

    • Cloud Deployment

    • Interactive Visualizations

    • CSV Export Support

    • Disease Gallery

    • Multi-Image Classification
    """)

    st.divider()

    st.subheader("🚀 Future Enhancements")

    st.write("""
    • Disease Severity Estimation

    • Treatment Recommendations

    • Explainable AI (Grad-CAM)

    • Transformer-Based Hybrid Models

    • Mobile Application

    • Hindi Language Support

    • Smart Agriculture Integration
    """)

    st.divider()

    st.success(
        "Thank you for using the AI Plant Disease Detection System 🌿"
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown(
    """
    <hr>

    <div class="footer">

    🌿 <b>AI Plant Disease Detection System</b><br>

    Built with TensorFlow • Streamlit • Plotly<br>

    Developed by <b>Kisalay Shukla</b>

    </div>
    """,
    unsafe_allow_html=True
)
