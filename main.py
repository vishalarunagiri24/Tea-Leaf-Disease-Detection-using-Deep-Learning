import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import io
import cv2
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet

# Load the model
try:
    model = tf.keras.models.load_model('best_vgg19.h5', custom_objects={'preprocess_input': preprocess_input})
except Exception as e:
    st.error(f"Failed to load the model 'best_vgg19.h5'. Ensure the file is in the project directory. Error: {str(e)}")
    st.stop()

class_names = ['algal_spot', 'brown_blight', 'gray_blight', 'healthy', 'helopeltis', 'red_spot']

# Disease insights
disease_insights = {
    'algal_spot': {
        'description': 'Caused by the algae Cephaleuros virescens, appearing as green to orange spots on leaves.',
        'symptoms': 'Velvety green or orange patches on upper leaf surfaces.',
        'management_tips': [
            'Improve air circulation by pruning tea bushes.',
            'Avoid overhead irrigation.',
            'Apply neem-based sprays.',
            'Ensure proper drainage.'
        ],
        'environmental_risks': 'High humidity and shaded conditions increase risk.'
    },
    'brown_blight': {
        'description': 'Caused by Colletotrichum species, leading to brown lesions.',
        'symptoms': 'Brown to black spots with gray-white centers.',
        'management_tips': [
            'Remove affected leaves.',
            'Apply mulch.',
            'Use resistant varieties.',
            'Maintain balanced fertilization.'
        ],
        'environmental_risks': 'Warm temperatures and rainfall promote spread.'
    },
    'gray_blight': {
        'description': 'Caused by Pestalotiopsis theae, causing grayish-white lesions.',
        'symptoms': 'Gray-white patches with dark borders.',
        'management_tips': [
            'Prune regularly.',
            'Avoid waterlogging.',
            'Use organic compost.',
            'Remove infected debris.'
        ],
        'environmental_risks': 'High humidity and poor air circulation.'
    },
    'helopeltis': {
        'description': 'Caused by the tea mosquito bug Helopeltis theivora.',
        'symptoms': 'Dark puncture marks with yellowing.',
        'management_tips': [
            'Introduce natural predators.',
            'Use sticky traps.',
            'Apply neem oil sprays.',
            'Maintain shade trees.'
        ],
        'environmental_risks': 'Warm, humid weather favors pest activity.'
    },
    'red_spot': {
        'description': 'Caused by fungi like Cercospora or rust pathogens.',
        'symptoms': 'Red or rust-like spots with yellow halos.',
        'management_tips': [
            'Ensure proper spacing.',
            'Avoid excessive watering.',
            'Use sulfur-based treatments.',
            'Rotate crops.'
        ],
        'environmental_risks': 'Warm, moist conditions with high humidity.'
    },
    'healthy': {
        'description': 'No signs of disease or pest damage.',
        'symptoms': 'Uniform green color and smooth texture.',
        'management_tips': [
            'Regular monitoring.',
            'Maintain soil fertility.',
            'Balanced irrigation.',
            'Prune periodically.'
        ],
        'environmental_risks': 'Monitor for changes in humidity or temperature.'
    }
}

# Helper functions omitted here for brevity - include them as-is from your previous code
# (segment_leaf, generate_gradcam_heatmap, apply_heatmap_threshold, overlay_heatmap,
# calculate_severity, pil_to_temp_file)

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
