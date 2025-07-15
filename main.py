import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import io, cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

# Load model
model = tf.keras.models.load_model('best_vgg19.h5', custom_objects={'preprocess_input': preprocess_input})
class_names = ['algal_spot','brown_blight','gray_blight','healthy','helopeltis','red_spot']

# Disease insights (Add your full dictionary)
disease_insights = { ... }

# Language support
LANGS = {'English': 'en', 'हिन्दी': 'hi'}
lang = st.sidebar.selectbox("Language", list(LANGS.keys()))
_ = lambda en, hi: hi if LANGS[lang]=='hi' else en

# Grad-CAM and utility functions
def generate_gradcam_heatmap(...): ...
def apply_heatmap_threshold(...): ...
def overlay_heatmap(...): ...
def calculate_severity(...): ...
def compute_health_score(conf, sev, roi_sev):
    return int((conf * 0.4 + (100 - sev) * 0.3 + (100 - roi_sev) * 0.3))
def treatment_advice(sev):
    if sev < 20:
        return _("Mild symptoms detected. Monitor regularly and apply basic organic sprays.", "हल्के लक्षण पाए गए। नियमित निगरानी करें और सामान्य जैविक स्प्रे का उपयोग करें।")
    elif sev < 50:
        return _("Moderate infection. Apply recommended fungicides and prune affected areas.", "मध्यम संक्रमण। अनुशंसित फफूंदनाशकों का प्रयोग करें और संक्रमित भागों की छंटाई करें।")
    else:
        return _("Severe infection detected. Immediate intervention and expert consultation advised.", "गंभीर संक्रमण पाया गया। तत्काल उपाय करें और विशेषज्ञ की सलाह लें।")

def pil_to_temp_file(image):
    temp_file = io.BytesIO()
    image.save(temp_file, format="PNG")
    temp_file.seek(0)
    return temp_file

def generate_pdf_report(predicted_class, confidence, severity, roi_sev, health_score, heatmap_image, original_image, insights):
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
    story.append(Paragraph(f"<b>ROI Severity Score:</b> {roi_sev:.2f}%", styles['Normal']))
    story.append(Paragraph(f"<b>Leaf Health Score:</b> {health_score}/100", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Visual Analysis", styles['Heading2']))
    story.append(Paragraph("Original Image", styles['Heading3']))
    original_img_file = pil_to_temp_file(original_image)
    story.append(RLImage(original_img_file, width=4*inch, height=2*inch))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Heatmap (Affected Areas)", styles['Heading3']))
    heatmap_img_file = pil_to_temp_file(heatmap_image)
    story.append(RLImage(heatmap_img_file, width=4*inch, height=2*inch))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Disease Insights", styles['Heading2']))
    story.append(Paragraph(f"<b>Description:</b> {insights.get('description', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Symptoms:</b> {insights.get('symptoms', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Environmental Risks:</b> {insights.get('environmental_risks', 'N/A')}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Management Tips", styles['Heading2']))
    for tip in insights.get('management_tips', ['No tips available.']):
        story.append(Paragraph(f"• {tip}", styles['Normal']))

    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

# UI Setup
st.set_page_config(page_title="Tea Leaf Disease Detection", layout="centered")
st.title(_('Tea Leaf Disease Detection','चाय के पत्ते की बीमारी की पहचान'))

# Image input
img_file = st.file_uploader("Upload Image", type=['jpg','png'])
cam_img = st.camera_input("Or capture using camera")
if img_file:
    image = Image.open(img_file).convert('RGB')
elif cam_img:
    image = Image.open(cam_img).convert('RGB')
else:
    st.stop()

st.image(image, caption=_("Input Image","इनपुट छवि"))

# Preprocess & Predict
img = image.resize((224,224))
arr = preprocess_input(np.array(img))
preds = model.predict(arr[np.newaxis])
cls = class_names[np.argmax(preds)]
conf = float(100 * np.max(preds))

# Heatmap & severity
heatmap = generate_gradcam_heatmap(arr[np.newaxis], model)
slider = st.slider("Heatmap Threshold", 10, 90, 60, step=5)
heatmap_thr = apply_heatmap_threshold(heatmap, slider)
overlay = overlay_heatmap(heatmap_thr, image)
st.image(overlay, caption=_("Heatmap Overlay","हीटमैप ओवरले"))

sev = calculate_severity(heatmap_thr)
st.write(f"{_('Severity','गंभीरता')}: {sev:.2f}%")

# ROI cropping
st.subheader(_('Select ROI for refined severity','ROI चयन करें'))
coords = st.selectbox(_('ROI Presets','ROI प्रीसेट'), ['Full image','Top-left','Center','Bottom-right'])
h, w = heatmap_thr.shape
if coords=='Top-left': roi = heatmap_thr[:h//2, :w//2]
elif coords=='Center': roi = heatmap_thr[h//4:3*h//4, w//4:3*w//4]
elif coords=='Bottom-right': roi = heatmap_thr[h//2:, w//2:]
else: roi = heatmap_thr
roi_sev = calculate_severity(roi)
st.write(f"{_('ROI Severity','ROI गंभीरता')}: {roi_sev:.2f}%")

# Health Score & Advice
hscore = compute_health_score(conf, sev, roi_sev)
st.subheader(_('Leaf Health Score','पत्ते की स्वास्थ्य स्कोर'))
st.progress(int(hscore))
st.write(f"{_('Score out of 100','100 में से स्कोर')}: {hscore}")

adv = treatment_advice(sev)
st.subheader(_('Recommended Action','अनुशंसित कार्रवाई'))
st.write(adv)

# Insights & Report
insights = disease_insights.get(cls, {})
pdf_data = generate_pdf_report(cls, conf, sev, roi_sev, hscore, overlay, image, insights)
st.download_button(
    label=_("Download PDF Report","पीडीएफ रिपोर्ट डाउनलोड करें"),
    data=pdf_data,
    file_name="tea_leaf_disease_report.pdf",
    mime="application/pdf"
)
st.info(_("Click the button to download your detailed PDF report.","अपनी विस्तृत पीडीएफ रिपोर्ट डाउनलोड करने के लिए बटन पर क्लिक करें।"))
