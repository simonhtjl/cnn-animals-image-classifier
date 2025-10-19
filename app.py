import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import io
import json
import pandas as pd
import zipfile
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt

st.set_page_config(page_title="CNN Animals Image Classifier", layout="wide")

@st.cache_resource
def load_keras_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def load_class_mapping(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = {int(k): v for k, v in data.items()}
        mapping = dict(sorted(mapping.items()))
        return mapping
    except Exception:
        return None

def get_model_input_shape(model):
    """Get the expected input shape from the model automatically"""
    try:
        if hasattr(model, 'layers') and len(model.layers) > 0:
            first_layer = model.layers[0]
            
            # Cara 1: Cek input_shape langsung
            if hasattr(first_layer, 'input_shape') and first_layer.input_shape is not None:
                input_shape = first_layer.input_shape
                if input_shape and len(input_shape) == 4:  # (batch, height, width, channels)
                    return (input_shape[1], input_shape[2])  # return (height, width)
            
            # Cara 2: Cek melalui config
            if hasattr(first_layer, 'get_config'):
                config = first_layer.get_config()
                if 'batch_input_shape' in config:
                    input_shape = config['batch_input_shape']
                    if input_shape and len(input_shape) == 4:
                        return (input_shape[1], input_shape[2])
            
            # Cara 3: Cek melalui model input
            if hasattr(model, 'input_shape') and model.input_shape is not None:
                input_shape = model.input_shape
                if input_shape and len(input_shape) == 4:
                    return (input_shape[1], input_shape[2])
                    
    except Exception as e:
        print(f"Error detecting input shape: {e}")
    
    return None

def preprocess_image(img: Image.Image, target_size):
    """Preprocess image to match model's expected input size"""
    img = img.convert("RGB")
    
    # Resize to target size
    img = img.resize(target_size)
    
    # Convert to array and normalize
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    return arr

def predict_image(model, img_arr, top_k=3):
    preds = model.predict(img_arr)[0]
    idxs = np.argsort(preds)[::-1][:top_k]
    return preds, idxs

def draw_bar_chart(labels, probs):
    fig, ax = plt.subplots(figsize=(6,3))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, probs[::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels[::-1])
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_xlim(0,1)
    st.pyplot(fig)

def save_batch_results(df, filename="predictions.csv"):
    return df.to_csv(index=False).encode('utf-8')

st.title("CNN Animals Image Classifier App")
st.markdown(
    """
    Aplikasi ini menerima gambar hewan dan menampilkan prediksi dari model CNN (Keras). 
    Fitur: upload model .h5, upload gambar tunggal, prediksi batch dari zip folder gambar (.zip), 
    serta mendownload hasil prediksi.
    """
)

# Sidebar controls
st.sidebar.header("Model & Settings")
default_model_path = "model/cnn_model.h5"
model_path = st.sidebar.text_input("Model file path (.h5)", value=default_model_path)
uploaded_model = st.sidebar.file_uploader("Atau upload model (.h5)", type=["h5"])

class_map_path = st.sidebar.text_input("class_indices.json path (opsional)", value="class_indices.json")
uploaded_class_map = st.sidebar.file_uploader("Upload class_indices.json (opsional)", type=["json"])

top_k = st.sidebar.slider("Top K predictions", min_value=1, max_value=10, value=3)

st.sidebar.markdown("---")
st.sidebar.markdown("Batch prediction")
zip_upload = st.sidebar.file_uploader("Upload ZIP file berisi gambar (opsional)", type=["zip"])
batch_confidence_threshold = st.sidebar.slider("Confidence threshold (for 'uncertain')", 0.0, 1.0, 0.5)

# Model loading area
model = None
class_mapping = None
model_loaded = False
model_input_shape = None
error_msg = None

# If user uploaded a model file, save to temp and load
if uploaded_model is not None:
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        tfile.write(uploaded_model.read())
        tfile.flush()
        uploaded_model_path = tfile.name
        model = load_keras_model(uploaded_model_path)
        model_loaded = True
        model_input_shape = get_model_input_shape(model)
        st.sidebar.success("Model uploaded & loaded.")
    except Exception as e:
        error_msg = f"Error loading uploaded model: {e}"
        st.sidebar.error(error_msg)
else:
    # try to load model from path
    if os.path.exists(model_path):
        try:
            model = load_keras_model(model_path)
            model_loaded = True
            model_input_shape = get_model_input_shape(model)
            st.sidebar.success(f"Loaded model: {model_path}")
        except Exception as e:
            error_msg = f"Error loading model at {model_path}: {e}"
            st.sidebar.error(error_msg)
    else:
        st.sidebar.info("Model file not found at path. Upload a .h5 model or place cnn_model.h5 in project folder.")

# Load class mapping if provided
if uploaded_class_map is not None:
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.write(uploaded_class_map.read())
        tmp.flush()
        class_mapping = load_class_mapping(tmp.name)
        if class_mapping:
            st.sidebar.success("Loaded class mapping from uploaded JSON.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded class mapping: {e}")
else:
    if os.path.exists(class_map_path):
        class_mapping = load_class_mapping(class_map_path)
        if class_mapping:
            st.sidebar.success(f"Loaded class mapping from {class_map_path}")

# Display model input shape info
if model_loaded and model_input_shape:
    st.sidebar.success(f"Model expects: {model_input_shape[0]}x{model_input_shape[1]} pixels")
elif model_loaded:
    st.sidebar.warning("Could not auto-detect input shape automatically")
    
    # Fallback: Manual detection from model summary
    try:
        # Get model summary as text
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        summary_text = "\n".join(summary_lines)
        
        # Look for input shape patterns in summary
        for line in summary_lines:
            if 'Input shape:' in line:
                # Extract shape from line like "Input shape: (None, 224, 224, 3)"
                import re
                shape_match = re.search(r'\(None, (\d+), (\d+), \d+\)', line)
                if shape_match:
                    height, width = int(shape_match.group(1)), int(shape_match.group(2))
                    model_input_shape = (height, width)
                    st.sidebar.success(f"Detected input shape: {height}x{width} (from summary)")
                    break
            elif 'conv2d input:' in line.lower():
                # Extract from conv layer line
                import re
                shape_match = re.search(r'\(None, (\d+), (\d+), \d+\)', line)
                if shape_match:
                    height, width = int(shape_match.group(1)), int(shape_match.group(2))
                    model_input_shape = (height, width)
                    st.sidebar.success(f"Detected input shape: {height}x{width} (from conv layer)")
                    break
        
        # If still not detected, use common sizes based on your training
        if not model_input_shape:
            # Since you trained with 224x224, force this size
            model_input_shape = (224, 224)
            st.sidebar.info("Using default input shape: 224x224 (from training)")
            
    except Exception as e:
        st.sidebar.error(f"Error in manual detection: {e}")
        # Ultimate fallback
        model_input_shape = (224, 224)
        st.sidebar.info("Forced input shape to 224x224")

# Main interactive area
col1, col2 = st.columns([1,1])

with col1:
    st.header("Upload Image of Animals for Prediction")
    uploaded_file = st.file_uploader("Upload image (jpg/png) - Any size accepted!", type=["jpg","jpeg","png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            original_size = image.size
            st.image(image, caption=f"Uploaded Image - Original size: {original_size[0]}x{original_size[1]}", use_container_width=True)
            
            if model_loaded:
                if model_input_shape:
                    # Use auto-detected model input shape
                    target_size = model_input_shape
                    st.success(f"Auto-resizing image to: {target_size[0]}x{target_size[1]} pixels")
                else:
                    # Fallback: try common sizes or use original size (not recommended)
                    st.warning("Could not detect model input shape. Trying common size 224x224")
                    target_size = (224, 224)
                
                arr = preprocess_image(image, target_size=target_size)
                preds, idxs = predict_image(model, arr, top_k=top_k)
                
                # build readable labels
                labels = []
                probs = []
                for i in idxs:
                    label = class_mapping.get(i, str(i)) if class_mapping else str(i)
                    labels.append(label)
                    probs.append(float(preds[i]))
                
                st.subheader("Predictions")
                for lbl, p in zip(labels, probs):
                    st.write(f"- {lbl} : {p*100:.2f}%")
                
                draw_bar_chart(labels, probs)
            else:
                st.warning("Model belum dimuat. Upload model (.h5) atau letakkan file .h5 di folder proyek dan masukkan path di sidebar.")
                
        except Exception as ex:
            st.error(f"Error processing image: {ex}")
            if "incompatible" in str(ex):
                st.info("Tip: This might be a model architecture issue. Try a different model or check the model summary in the right panel.")

with col2:
    st.header("Model Information")
    if model_loaded:
        try:
            st.write("Model Architecture:")
            # show light summary
            text = []
            model.summary(print_fn=lambda x: text.append(x))
            summary_str = "\n".join(text)
            st.text_area("Model Summary", value=summary_str, height=300)
            
            if model_input_shape:
                st.success(f"Expected input shape: {model_input_shape[0]}x{model_input_shape[1]}x3")
            else:
                st.warning("Could not auto-detect input shape")
                
        except Exception:
            st.write("Could not render model summary.")
    else:
        st.info("Model belum dimuat. Gunakan panel sidebar untuk memilih/upload model.")

# Batch prediction area
st.markdown("---")
st.header("Batch Prediction (ZIP of images)")
st.markdown("Upload file .zip yang berisi gambar (jpg/png) - semua ukuran diterima! Hasil akan di-download sebagai CSV.")

if zip_upload is not None:
    if not model_loaded:
        st.warning("Model belum dimuat — batch prediction membutuhkan model.")
    else:
        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_upload.read()))
            members = [m for m in zf.namelist() if not m.endswith('/') and Path(m).suffix.lower() in {'.jpg','.jpeg','.png'}]
            
            if len(members) == 0:
                st.error("ZIP tidak berisi gambar jpg/png.")
            else:
                st.write(f"Found {len(members)} images. Running predictions...")
                
                # Determine target size for batch processing
                if model_input_shape:
                    batch_target_size = model_input_shape
                else:
                    batch_target_size = (224, 224)  # fallback
                    
                st.info(f"Resizing all images to: {batch_target_size[0]}x{batch_target_size[1]}")
                
                rows = []
                progress_bar = st.progress(0)
                
                for i, m in enumerate(members):
                    try:
                        data = zf.read(m)
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        original_size = img.size
                        
                        # Resize to model's expected size
                        arr = preprocess_image(img, target_size=batch_target_size)
                        preds = model.predict(arr, verbose=0)[0]
                        idx = int(np.argmax(preds))
                        prob = float(np.max(preds))
                        label = class_mapping.get(idx, str(idx)) if class_mapping else str(idx)
                        uncertain = prob < batch_confidence_threshold
                        
                        rows.append({
                            "filename": m, 
                            "original_size": f"{original_size[0]}x{original_size[1]}",
                            "resized_to": f"{batch_target_size[0]}x{batch_target_size[1]}",
                            "pred_idx": idx, 
                            "pred_label": label, 
                            "confidence": prob, 
                            "uncertain": uncertain
                        })
                    except Exception as e:
                        rows.append({
                            "filename": m, 
                            "original_size": "N/A",
                            "resized_to": "N/A",
                            "pred_idx": None, 
                            "pred_label": None, 
                            "confidence": None, 
                            "uncertain": None, 
                            "error": str(e)
                        })
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(members))
                
                df = pd.DataFrame(rows)
                st.dataframe(df.head(100))
                
                csv_bytes = save_batch_results(df)
                st.download_button(
                    "Download predictions CSV", 
                    data=csv_bytes, 
                    file_name="predictions.csv", 
                    mime="text/csv"
                )
                
        except zipfile.BadZipFile:
            st.error("File yang diupload bukan ZIP yang valid.")

# Small utilities / tips
st.markdown("---")
st.header("Tips & Notes")
st.markdown(
    """
    - Auto-resizing: Sekarang aplikasi secara otomatis mendeteksi ukuran input yang diharapkan model dan menyesuaikan gambar Anda
    - Untuk mapping index->class, simpan file class_indices.json saat training: json.dump({str(v):k for k,v in train_gen.class_indices.items()}, open('class_indices.json','w'))
    - Jika model tidak bisa dimuat, periksa versi TensorFlow & Python (disarankan Python 3.10/3.11)
    - Aplikasi ini menerima gambar ukuran berapa pun dan akan otomatis menyesuaikan dengan kebutuhan model
    """
)

# Footer: optional sample images (if folder 'sample_images' exists)
sample_dir = "sample_images"
if os.path.exists(sample_dir) and os.path.isdir(sample_dir):
    st.markdown("---")
    st.subheader("Sample images (from sample_images folder)")
    files = [f for f in os.listdir(sample_dir) if Path(f).suffix.lower() in {'.jpg','.jpeg','.png'}]
    cols = st.columns(4)
    for i, fname in enumerate(files[:12]):
        img = Image.open(os.path.join(sample_dir, fname))
        with cols[i % 4]:
            st.image(img, caption=f"{fname} ({img.size[0]}x{img.size[1]})", use_container_width=True)

st.markdown("---")
st.markdown(""" Anggota Kelompok 6:
- Dini Ambarwati — 2802647783
- Fakhrusy Hassan Siregar — 2802642284
- Najdi Fadhlur Rahman — 2802519625
- Simon Mangasi Hutajulu — 2802647373
- Sujud Hosis Sudarja — 2802633172 """)

