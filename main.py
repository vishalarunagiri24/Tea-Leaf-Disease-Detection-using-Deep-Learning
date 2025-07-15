import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import io
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

try:
    model = tf.keras.models.load_model('best_vgg19.h5', custom_objects={'preprocess_input': preprocess_input})
except Exception as e:
    st.error(f"Failed to load the model 'best_vgg19.h5'. Ensure the file is in the project directory. Error: {str(e)}")
    st.stop()

class_names = ['algal_spot', 'brown_blight', 'gray_blight', 'healthy', 'helopeltis', 'red_spot']

disease_insights = {
    'algal_spot': {
        'description': 'Caused by the algae *Cephaleuros virescens*, appearing as green to orange spots on leaves. Thrives in humid, shaded conditions.',
        'symptoms': 'Velvety green or orange patches on upper leaf surfaces, often with a powdery texture.',
        'management_tips': [
            'Improve air circulation by pruning tea bushes.',
            'Avoid overhead irrigation to reduce leaf wetness.',
            'Apply neem-based organic sprays to suppress algae growth.',
            'Ensure proper drainage to reduce humidity.'
        ],
        'environmental_risks': 'High humidity (>80%) and shaded conditions increase risk. Avoid excessive moisture.'
    },
    'brown_blight': {
        'description': 'Caused by *Colletotrichum* species, a fungal disease leading to brown lesions. Common in warm, wet climates.',
        'symptoms': 'Brown to black spots with gray-white centers, often on older leaves, leading to defoliation.',
        'management_tips': [
            'Remove and destroy affected leaves to reduce fungal spores.',
            'Apply mulch to prevent soil splashing onto leaves.',
            'Use resistant tea varieties if available.',
            'Maintain balanced fertilization to avoid excessive nitrogen.'
        ],
        'environmental_risks': 'Warm temperatures (20-30°C) and frequent rainfall promote spread. Reduce leaf wetness.'
    },
    'gray_blight': {
        'description': 'Caused by *Pestalotiopsis theae*, a fungus causing grayish-white lesions. Prevalent in high-humidity areas.',
        'symptoms': 'Gray-white patches with dark borders, often on leaf margins, causing leaf drop.',
        'management_tips': [
            'Prune regularly to improve ventilation.',
            'Avoid waterlogging by improving soil drainage.',
            'Use organic compost to boost plant immunity.',
            'Monitor and remove infected debris promptly.'
        ],
        'environmental_risks': 'High humidity and poor air circulation exacerbate this disease. Ensure proper ventilation.'
    },
    'helopeltis': {
        'description': 'Caused by the tea mosquito bug (*Helopeltis theivora*), an insect pest that sucks sap, damaging young leaves and shoots.',
        'symptoms': 'Small, dark puncture marks with yellowing around feeding sites, leading to curled or distorted leaves.',
        'management_tips': [
            'Introduce natural predators like spiders or ladybugs.',
            'Use sticky traps to monitor and reduce pest populations.',
            'Apply neem oil sprays during early pest detection.',
            'Maintain shade trees to disrupt pest breeding.'
        ],
        'environmental_risks': 'Warm, humid weather (25-30°C) favors pest activity. Monitor during monsoon seasons.'
    },
    'red_spot': {
        'description': 'Caused by fungi like *Cercospora* or rust pathogens, appearing as red to rust-colored spots. Common in warm, moist conditions.',
        'symptoms': 'Reddish or rust-like spots scattered on leaves, sometimes with yellow halos, reducing photosynthesis.',
        'management_tips': [
            'Ensure proper spacing between tea plants for airflow.',
            'Avoid excessive watering to keep foliage dry.',
            'Use sulfur-based organic treatments for early control.',
            'Rotate crops or intercrop to reduce fungal buildup.'
        ],
        'environmental_risks': 'Warm, moist conditions (20-28°C) with high humidity increase risk. Keep foliage dry.'
    },
    'healthy': {
        'description': 'The leaf shows no signs of disease or pest damage, indicating good plant health.',
        'symptoms': 'Uniform green color, smooth texture, no spots or lesions.',
        'management_tips': [
            'Continue regular monitoring for early pest or disease detection.',
            'Maintain soil fertility with organic compost.',
            'Ensure balanced irrigation to avoid stress.',
            'Prune periodically to promote healthy growth.'
        ],
        'environmental_risks': 'No immediate risks, but monitor for sudden changes in humidity or temperature.'
    }
}

def segment_leaf(image_np):
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray, dtype=np.uint8)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(image_np, image_np, mask=mask)
        return masked_image, mask
    except Exception as e:
        st.warning(f"Failed to segment leaf: {str(e)}. Using original image.")
        return image_np, np.ones_like(image_np[:, :, 0], dtype=np.uint8) * 255

def generate_gradcam_heatmap(img_array, model, last_conv_layer_name='block3_conv4'):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            predicted_class = tf.argmax(predictions[0])
            class_output = predictions[:, predicted_class]
        grads = tape.gradient(class_output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-10
        return heatmap
    except Exception as e:
        st.warning(f"Failed to generate heatmap: {str(e)}. Displaying prediction without heatmap.")
        return None

def apply_heatmap_threshold(heatmap, percentile):
    threshold = np.percentile(heatmap, percentile)
    heatmap = np.where(heatmap >= threshold, heatmap, 0)
    return heatmap

def overlay_heatmap(heatmap, original_image, mask, alpha=0.8):
    try:
        if isinstance(original_image, Image.Image):
            original_image_np = np.array(original_image)
        else:
            original_image_np = original_image
        heatmap = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))
        mask = cv2.resize(mask, (original_image_np.shape[1], original_image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.uint8)
        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        heatmap = np.uint8(255 * (heatmap / (np.max(heatmap) + 1e-10)))
        heatmap_colored = np.zeros_like(original_image_np)
        mask_heatmap = heatmap > 0
        heatmap_colored[mask_heatmap] = [255, 0, 0]
        superimposed_img = cv2.addWeighted(original_image_np, 1 - alpha, heatmap_colored, alpha, 0.0)
        return Image.fromarray(superimposed_img)
    except Exception as e:
        st.warning(f"Failed to overlay heatmap: {str(e)}. Returning original image.")
        return original_image

def calculate_severity(heatmap, mask):
    try:
        heatmap = cv2.resize(heatmap, (mask.shape[1], mask.shape[0]))
        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask.astype(np.uint8))
        leaf_area = np.sum(mask > 0)
        affected_area = np.sum(heatmap > 0)
        severity = (affected_area / leaf_area) * 100 if leaf_area > 0 else 0
        return min(severity, 100)
    except Exception as e:
        st.warning(f"Failed to calculate severity: {str(e)}. Defaulting to 0%.")
        return 0

def pil_to_temp_file(image):
    temp_file = io.BytesIO()
    image.save(temp_file, format="PNG")
    temp_file.seek(0)
    return temp_file

def generate_pdf_report(predicted_class, confidence, severity, heatmap_image, original_image, insights):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    current_date = datetime.now().strftime("%B %d, %Y")
    story.append(Paragraph("Tea Leaf Disease Analysis Report", styles['Title']))
    story.append(Paragraph(f"Generated on {current_date}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Prediction Summary", styles['Heading2']))
    story.append(Paragraph(f"<b>Disease:</b> {predicted_class.replace('_', ' ').title()}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']))
    story.append(Paragraph(f"<b>Severity Score:</b> {severity:.2f}%", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Visual Analysis", styles['Heading2']))
    story.append(Paragraph("Original Image", styles['Heading3']))
    original_img_file = pil_to_temp_file(original_image)
    story.append(ReportLabImage(original_img_file, width=4*inch, height=2*inch))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Heatmap (Affected Areas)", styles['Heading3']))
    heatmap_img_file = pil_to_temp_file(heatmap_image)
    story.append(ReportLabImage(heatmap_img_file, width=4*inch, height=2*inch))
    story.append(Paragraph("Red areas indicate regions of the leaf most affected by the disease.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Disease Insights", styles['Heading2']))
    story.append(Paragraph(f"<b>Description:</b> {insights.get('description', 'No information available.')}", styles['Normal']))
    story.append(Paragraph(f"<b>Symptoms:</b> {insights.get('symptoms', 'No symptoms available.')}", styles['Normal']))
    story.append(Paragraph(f"<b>Environmental Risks:</b> {insights.get('environmental_risks', 'No risks available.')}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Management Tips", styles['Heading2']))
    for tip in insights.get('management_tips', ['No tips available.']):
        story.append(Paragraph(f"• {tip}", styles['Normal']))
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

# Streamlit UI
st.set_page_config(page_title="Tea Leaf Disease Classification", layout="centered")
st.title("Tea Leaf Disease Detection using VGG19")
st.write("Upload an image of a tea leaf to predict the disease, analyze its severity, view a heatmap of affected areas, and get management tips.")

uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    img = image.resize((224, 224))
    img_array = np.array(img)

    with st.spinner('Segmenting leaf...'):
        masked_image, mask = segment_leaf(img_array)

    img_array_preprocessed = preprocess_input(img_array.copy())
    img_array_preprocessed = np.expand_dims(img_array_preprocessed, axis=0)

    with st.spinner('Predicting...'):
        prediction = model.predict(img_array_preprocessed)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class.replace('_', ' ').title()} ({confidence:.2f}% confidence)")

    st.subheader("Prediction Probabilities:")
    probs = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    st.bar_chart(probs)

    with st.spinner('Generating heatmap...'):
        heatmap = generate_gradcam_heatmap(img_array_preprocessed, model)
        if heatmap is not None:
            st.subheader("Heatmap of Affected Areas:")
            threshold_percentile = st.slider("Adjust Heatmap Threshold (Show top X% of focus areas)", 10, 90, 60, step=5)
            heatmap_adjusted = apply_heatmap_threshold(heatmap.copy(), threshold_percentile)
            st.subheader("Raw Heatmap (Grayscale):")
            heatmap_normalized = np.uint8(255 * (heatmap_adjusted / (np.max(heatmap_adjusted) + 1e-10)))
            st.image(heatmap_normalized, caption='Raw heatmap (brighter areas = higher focus)', use_container_width=True, clamp=True)
            superimposed_img = overlay_heatmap(heatmap_adjusted, image, mask)
            st.image(superimposed_img, caption='Heatmap Overlay (Red areas indicate model focus)', use_container_width=True)
            severity = calculate_severity(heatmap_adjusted, mask)
            st.subheader("Disease Severity Score:")
            st.write(f"The disease affects approximately {severity:.2f}% of the leaf area.")
        else:
            st.image(image, caption='Original image (heatmap generation failed)', use_container_width=True)
            severity = 0

    st.subheader("Disease Insights:")
    insights = disease_insights.get(predicted_class, {})
    st.markdown(f"**Description**: {insights.get('description', 'No information available.')}")
    st.markdown(f"**Symptoms**: {insights.get('symptoms', 'No symptoms available.')}")

    st.subheader("Environmental Risk Alerts:")
    st.markdown(f"{insights.get('environmental_risks', 'No risks available.')}")

    st.subheader("Management Tips:")
    for tip in insights.get('management_tips', ['No tips available.']):
        st.markdown(f"- {tip}")

    st.subheader("Download Report:")
    if heatmap is not None:
        pdf_data = generate_pdf_report(predicted_class, confidence, severity, superimposed_img, image, insights)
        st.download_button(
            label="Download PDF Report",
            data=pdf_data,
            file_name="tea_leaf_disease_report.pdf",
            mime="application/pdf"
        )
        st.info("Click the button to download a directly viewable PDF report.")
    else:
        st.warning("Cannot generate report without a heatmap.")
