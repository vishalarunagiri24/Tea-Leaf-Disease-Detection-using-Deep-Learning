import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set matplotlib backend to Agg for Streamlit compatibility
import io
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
# Check for required dependencies
# try:
#     import cv2
# except ImportError:
#     st.error("OpenCV (cv2) is not installed. Please install it using: `pip install opencv-python`")
#     st.stop()

# try:
#     import matplotlib.pyplot as plt
# except ImportError:
#     st.error("Matplotlib is not installed. Please install it using: `pip install matplotlib`")
#     st.stop()

# try:
#     from reportlab.pdfgen import canvas
#     from reportlab.lib.pagesizes import A4
#     from reportlab.lib.units import inch
#     from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
#     from reportlab.lib.styles import getSampleStyleSheet
# except ImportError:
#     st.error("ReportLab is not installed. Please install it using: `pip install reportlab`")
#     st.stop()

# Load the model
try:
    model = tf.keras.models.load_model('best_vgg19.h5', custom_objects={'preprocess_input': preprocess_input})
except Exception as e:
    st.error(f"Failed to load the model 'best_vgg19.h5'. Ensure the file is in the project directory. Error: {str(e)}")
    st.stop()

class_names = ['algal_spot', 'brown_blight', 'gray_blight', 'healthy', 'helopeltis', 'red_spot']

# Disease insights, management tips, and environmental risks
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

# Function to segment the leaf from the background
def segment_leaf(image_np):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        # Apply thresholding to create a binary mask
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create a mask for the largest contour (assumed to be the leaf)
        mask = np.zeros_like(gray, dtype=np.uint8)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(image_np, image_np, mask=mask)
        return masked_image, mask
    except Exception as e:
        st.warning(f"Failed to segment leaf: {str(e)}. Using original image.")
        return image_np, np.ones_like(image_np[:, :, 0], dtype=np.uint8) * 255

# Function to generate Grad-CAM heatmap
def generate_gradcam_heatmap(img_array, model, last_conv_layer_name='block3_conv4'):
    try:
        # Create a model that outputs the last conv layer and the final predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute gradients of the top predicted class with respect to the conv layer output
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            predicted_class = tf.argmax(predictions[0])
            class_output = predictions[:, predicted_class]
        
        # Get gradients and conv outputs
        grads = tape.gradient(class_output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-10  # Avoid division by zero
        
        return heatmap
    except Exception as e:
        st.warning(f"Failed to generate heatmap: {str(e)}. Displaying prediction without heatmap.")
        return None

# Function to apply threshold to heatmap
def apply_heatmap_threshold(heatmap, percentile):
    threshold = np.percentile(heatmap, percentile)
    heatmap = np.where(heatmap >= threshold, heatmap, 0)
    return heatmap

# Function to overlay heatmap on the original image
def overlay_heatmap(heatmap, original_image, mask, alpha=0.8):
    try:
        # Convert PIL Image to NumPy array if necessary
        if isinstance(original_image, Image.Image):
            original_image_np = np.array(original_image)
        else:
            original_image_np = original_image
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))
        
        # Resize mask to match original image size (if not already)
        mask = cv2.resize(mask, (original_image_np.shape[1], original_image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.uint8)  # Ensure mask is uint8
        
        # Debug: Log shapes and types
        st.write(f"Heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")
        st.write(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        
        # Apply mask to heatmap (focus only on leaf)
        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)
        
        # Apply Gaussian blur to reduce noise
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Normalize heatmap for visualization
        heatmap = np.uint8(255 * (heatmap / (np.max(heatmap) + 1e-10)))
        
        # Create a binary colormap: red for high focus, transparent for low focus
        heatmap_colored = np.zeros_like(original_image_np)
        mask_heatmap = heatmap > 0
        heatmap_colored[mask_heatmap] = [255, 0, 0]  # Red for high focus areas
        
        # Overlay heatmap
        superimposed_img = original_image_np.copy()
        superimposed_img = cv2.addWeighted(
            superimposed_img, 1 - alpha, heatmap_colored, alpha, 0.0
        )
        
        return Image.fromarray(superimposed_img)
    except Exception as e:
        st.warning(f"Failed to overlay heatmap: {str(e)}. Returning original image.")
        return original_image

# Function to calculate disease severity based on heatmap
def calculate_severity(heatmap, mask):
    try:
        # Resize heatmap to match mask size if necessary
        heatmap = cv2.resize(heatmap, (mask.shape[1], mask.shape[0]))
        # Apply mask to heatmap
        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask.astype(np.uint8))
        # Calculate the proportion of the leaf area covered by the heatmap
        leaf_area = np.sum(mask > 0)
        affected_area = np.sum(heatmap > 0)
        severity = (affected_area / leaf_area) * 100 if leaf_area > 0 else 0
        return min(severity, 100)  # Cap at 100%
    except Exception as e:
        st.warning(f"Failed to calculate severity: {str(e)}. Defaulting to 0%.")
        return 0

# Function to save PIL image to a temporary file for reportlab
def pil_to_temp_file(image):
    temp_file = io.BytesIO()
    image.save(temp_file, format="PNG")
    temp_file.seek(0)
    return temp_file

# Function to generate a PDF report using reportlab
def generate_pdf_report(predicted_class, confidence, severity, heatmap_image, original_image, insights):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Current Date 
    current_date = datetime.now().strftime("%B %d, %Y")

    
    # Title
    story.append(Paragraph("Tea Leaf Disease Analysis Report", styles['Title']))
    story.append(Paragraph(f"Generated on {current_date}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Prediction Summary
    story.append(Paragraph("Prediction Summary", styles['Heading2']))
    story.append(Paragraph(f"<b>Disease:</b> {predicted_class.replace('_', ' ').title()}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']))
    story.append(Paragraph(f"<b>Severity Score:</b> {severity:.2f}%", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Visual Analysis
    story.append(Paragraph("Visual Analysis", styles['Heading2']))

    # Original Image
    story.append(Paragraph("Original Image", styles['Heading3']))
    original_img_file = pil_to_temp_file(original_image)
    story.append(ReportLabImage(original_img_file, width=4*inch, height=2*inch))
    story.append(Spacer(1, 0.1 * inch))

    # Heatmap Image
    story.append(Paragraph("Heatmap (Affected Areas)", styles['Heading3']))
    heatmap_img_file = pil_to_temp_file(heatmap_image)
    story.append(ReportLabImage(heatmap_img_file, width=4*inch, height=2*inch))
    story.append(Paragraph("Red areas indicate regions of the leaf most affected by the disease.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Disease Insights
    story.append(Paragraph("Disease Insights", styles['Heading2']))
    story.append(Paragraph(f"<b>Description:</b> {insights.get('description', 'No information available.')}", styles['Normal']))
    story.append(Paragraph(f"<b>Symptoms:</b> {insights.get('symptoms', 'No symptoms available.')}", styles['Normal']))
    story.append(Paragraph(f"<b>Environmental Risks:</b> {insights.get('environmental_risks', 'No risks available.')}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Management Tips
    story.append(Paragraph("Management Tips", styles['Heading2']))
    for tip in insights.get('management_tips', ['No tips available.']):
        story.append(Paragraph(f"• {tip}", styles['Normal']))

    # Build the PDF
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

# Streamlit UI
st.set_page_config(page_title="Tea Leaf Disease Classification", layout="centered")
st.title("🍃 Tea Leaf Disease Detection using VGG19")
st.write("Upload an image of a tea leaf to predict the disease, analyze its severity, view a heatmap of affected areas, and get management tips!")

# File uploader for image
uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocessing the image for prediction
    img = image.resize((224, 224))  # Resize to VGG19 input size
    img_array = np.array(img)
    
    # Segment the leaf
    with st.spinner('Segmenting leaf...'):
        masked_image, mask = segment_leaf(img_array)
        st.image(masked_image, caption='Segmented Leaf', use_container_width=True)
    
    # Preprocess the image for VGG19
    img_array_preprocessed = preprocess_input(img_array.copy())  # Copy to avoid modifying original
    img_array_preprocessed = np.expand_dims(img_array_preprocessed, axis=0)  # Add batch dimension: (1, 224, 224, 3)
    
    # Make the prediction
    with st.spinner('Predicting...'):
        prediction = model.predict(img_array_preprocessed)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    # Display the prediction result
    st.success(f"**Prediction:** {predicted_class.replace('_', ' ').title()} ({confidence:.2f}% confidence)")

    # Display the prediction probabilities
    st.subheader("🔎 Prediction Probabilities:")
    probs = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    st.bar_chart(probs)

    # Generate Grad-CAM heatmap
    with st.spinner('Generating heatmap...'):
        heatmap = generate_gradcam_heatmap(img_array_preprocessed, model)
        if heatmap is not None:
            # Interactive heatmap adjustment
            st.subheader("🗺️ Heatmap of Affected Areas:")
            threshold_percentile = st.slider("Adjust Heatmap Threshold (Show top X% of focus areas)", 10, 90, 60, step=5)
            heatmap_adjusted = apply_heatmap_threshold(heatmap.copy(), threshold_percentile)
            
            # Display raw heatmap for debugging
            st.subheader("🔍 Raw Heatmap (Grayscale):")
            heatmap_normalized = np.uint8(255 * (heatmap_adjusted / (np.max(heatmap_adjusted) + 1e-10)))
            st.image(heatmap_normalized, caption='Raw heatmap (brighter areas = higher focus)', use_container_width=True, clamp=True)
            
            # Overlay heatmap on original image
            superimposed_img = overlay_heatmap(heatmap_adjusted, image, mask)
            st.image(superimposed_img, caption='Heatmap Overlay (Red areas indicate model focus)', use_container_width=True)
            
            # Calculate and display severity score
            severity = calculate_severity(heatmap_adjusted, mask)
            st.subheader("📊 Disease Severity Score:")
            st.write(f"The disease affects approximately **{severity:.2f}%** of the leaf area.")
        else:
            st.image(image, caption='Original image (heatmap generation failed)', use_container_width=True)
            severity = 0

    # Display disease insights and management tips
    st.subheader("ℹ️ Disease Insights:")
    insights = disease_insights.get(predicted_class, {})
    st.markdown(f"**Description**: {insights.get('description', 'No information available.')}")
    st.markdown(f"**Symptoms**: {insights.get('symptoms', 'No symptoms available.')}")
    
    # Display environmental risks
    st.subheader("⚠️ Environmental Risk Alerts:")
    st.markdown(f"{insights.get('environmental_risks', 'No risks available.')}")
    
    st.subheader("🌱 Management Tips:")
    tips = insights.get('management_tips', ['No tips available.'])
    for tip in tips:
        st.markdown(f"- {tip}")

    # Generate and provide downloadable PDF report
    st.subheader("📄 Download Report:")
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
