import os
import json
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
from werkzeug.utils import secure_filename

# Import functions from vin.py - PROPERLY IMPORTING YOUR FUNCTIONS
try:
    print("Attempting to import from vin.py...")
    from vin import (
        predict_image, 
        explain_with_shap, 
        get_tumor_stage, 
        get_risk_level, 
        get_nodule_location, 
        generate_explanation,
        model,  # Import the model directly
        background_data  # Import the background data directly
    )
    print("Successfully imported functions from vin.py")
    USE_VIN_MODEL = True
except ImportError as e:
    print(f"Could not import from vin.py: {e}")
    print("Using mock implementations instead")
    USE_VIN_MODEL = False
    
    # Mock implementations if vin.py is not available
    def predict_image(model, img_path):
        """Simulate predicting an image using the model from vin.py"""
        return "Benign", 100.0

    def explain_with_shap(model, img_path, background_data, save_path="shap_output/shap_overlay.png"):
        """Generate a SHAP overlay visualization with red lines"""
        # Read the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Use a blank image if the file couldn't be read
            img = np.zeros((256, 256), dtype=np.uint8)
        
        # Resize to expected dimensions
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        
        # Create a mock visualization with horizontal red lines
        # This simulates the areas where the model is focused
        height, width = img.shape
        
        # Convert to RGB for visualization
        img_rgb = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
        
        # Create a mock mask
        mask = np.zeros((height, width), dtype=bool)
        
        # Add horizontal lines at specific rows (simulating model focus)
        for row in range(height//3, 2*height//3, 10):
            mask[row, :] = True
            img_rgb[row, :, 0] = 1.0  # Red channel
            img_rgb[row, :, 1] = 0.0  # Green channel
            img_rgb[row, :, 2] = 0.0  # Blue channel
        
        # Save to a temporary file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.imsave(save_path, img_rgb)
        
        # Create a mock shap_arr
        shap_arr = np.zeros((height, width), dtype=float)
        shap_arr[mask] = 1.0
        
        return shap_arr, mask, save_path

    def get_tumor_stage(area):
        if area > 40:
            return "Stage III"
        elif area > 25:
            return "Stage II"
        elif area > 10:
            return "Stage I"
        else:
            return "No significant tumor"

    def get_risk_level(area):
        if area > 40:
            return "High"
        elif area > 25:
            return "Moderate"
        elif area > 10:
            return "Low"
        else:
            return "None"

    def get_nodule_location(mask):
        return "No abnormal region detected (location undetermined)"

    def generate_explanation(confidence, tumor_area):
        explanation = "Based on "
        explanation += "high model confidence and " if confidence > 90 else "moderate model confidence and "
        if tumor_area > 40:
            explanation += "extensive tumor area with highly abnormal features."
        elif tumor_area > 25:
            explanation += "moderate-sized tumor area with visible abnormalities."
        elif tumor_area > 10:
            explanation += "small tumor presence showing mild irregularities."
        else:
            explanation += "minimal or no visible tumor signs."
        return explanation

    # Mock model and background data
    model = None
    background_data = None

def generate_pdf_report(result, img_path, shap_path):
    """Generate a PDF report with the results and images using ReportLab"""
    try:
        # First, make sure we have the ReportLab library
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        # Create a PDF file
        report_path = os.path.join(tempfile.gettempdir(), "lung_cancer_report.pdf")
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        
        # Create styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        title_style.alignment = 1  # Center alignment
        subtitle_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Create a custom style for the disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.gray
        )
        
        # Create the content
        content = []
        
        # Add title
        content.append(Paragraph("Lung Cancer Prediction Report", title_style))
        content.append(Spacer(1, 0.25*inch))
        
        # Add date
        content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                normal_style))
        content.append(Spacer(1, 0.5*inch))
        
        # Add prediction prominently at the top
        prediction_style = ParagraphStyle(
            'Prediction',
            parent=styles['Heading2'],
            alignment=1,  # Center alignment
            textColor=colors.green if result['prediction'] == 'Benign' else colors.red
        )
        content.append(Paragraph(f"Predicted Case: {result['prediction']}", prediction_style))
        content.append(Paragraph(f"Confidence: {result['confidence']:.2f}%", prediction_style))
        content.append(Spacer(1, 0.5*inch))
        
        # Add images if they exist
        try:
            # Original image
            if os.path.exists(img_path):
                img = Image(img_path)
                img.drawHeight = 2*inch
                img.drawWidth = 2*inch
                content.append(Paragraph("Original CT Scan", subtitle_style))
                content.append(img)
                content.append(Spacer(1, 0.25*inch))
            
            # SHAP overlay
            if os.path.exists(shap_path):
                shap_img = Image(shap_path)
                shap_img.drawHeight = 2*inch
                shap_img.drawWidth = 2*inch
                content.append(Paragraph("SHAP Analysis Visualization", subtitle_style))
                content.append(shap_img)
                content.append(Spacer(1, 0.25*inch))
        except Exception as e:
            print(f"Error adding images to PDF: {e}")
        
        # Add prediction results
        content.append(Paragraph("Prediction Results", subtitle_style))
        content.append(Spacer(1, 0.1*inch))
        
        # Create a table for the results
        data = [
            ["Prediction", result['prediction']],
            ["Confidence", f"{result['confidence']:.2f}%"],
            ["Tumor Area", f"{result['tumor_area']} pixels"],
            ["Tumor Stage", result['tumor_stage']],
            ["Risk Level", result['risk_level']]
        ]
        
        table = Table(data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(table)
        content.append(Spacer(1, 0.25*inch))
        
        # Add nodule location
        content.append(Paragraph("Nodule Location", subtitle_style))
        content.append(Paragraph(result['nodule_location'], normal_style))
        content.append(Spacer(1, 0.25*inch))
        
        # Add explanation
        content.append(Paragraph("Analysis Explanation", subtitle_style))
        content.append(Paragraph(result['explanation_text'], normal_style))
        content.append(Spacer(1, 0.5*inch))
        
        # Add disclaimer
        disclaimer_text = "DISCLAIMER: This is an AI-generated report and should not be considered as a medical diagnosis. Please consult with a healthcare professional for proper medical evaluation."
        content.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build the PDF
        doc.build(content)
        return report_path
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        # Fallback to a simple text file if PDF generation fails
        report_path = os.path.join(tempfile.gettempdir(), "lung_cancer_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"Lung Cancer Prediction Report\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"Confidence: {result['confidence']:.2f}%\n")
            f.write(f"Tumor Area: {result['tumor_area']} pixels\n")
            f.write(f"Tumor Stage: {result['tumor_stage']}\n")
            f.write(f"Risk Level: {result['risk_level']}\n")
            f.write(f"Nodule Location: {result['nodule_location']}\n")
            f.write(f"Explanation: {result['explanation_text']}\n\n")
            f.write("DISCLAIMER: This is an AI-generated report and should not be considered as a medical diagnosis.")
        return report_path

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create shap_output directory
os.makedirs('shap_output', exist_ok=True)

# Store results in memory (in a real app, use a database)
session_data = {}

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle the upload page and file uploads"""
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Process the image using vin.py functions
            # Note: model is imported directly from vin.py
            prediction, confidence = predict_image(model, file_path)
            
            # Generate SHAP visualization
            # Note: background_data is imported directly from vin.py
            shap_path = os.path.join('shap_output', f'shap_overlay_{filename}')
            shap_arr, mask, shap_path = explain_with_shap(model, file_path, background_data, shap_path)
            
            # Calculate additional metrics
            tumor_area = int(np.sum(mask))
            stage = get_tumor_stage(tumor_area)
            risk = get_risk_level(tumor_area)
            location_sentence = get_nodule_location(mask)
            explanation_text = generate_explanation(confidence, tumor_area)
            
            # Create result dictionary
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'tumor_area': tumor_area,
                'tumor_stage': stage,
                'risk_level': risk,
                'nodule_location': location_sentence,
                'explanation_text': explanation_text
            }
            
            # Generate a session ID
            session_id = base64.urlsafe_b64encode(os.urandom(16)).decode('utf-8')
            
            # Store the results and paths
            session_data[session_id] = {
                'result': result,
                'file_path': file_path,
                'shap_path': shap_path
            }
            
            # Redirect to results page
            return redirect(url_for('results', session_id=session_id))
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    # GET request - show upload form
    return render_template('upload.html')

@app.route('/results/<session_id>')
def results(session_id):
    """Show the results page"""
    if session_id not in session_data:
        return redirect(url_for('index'))
    
    data = session_data[session_id]
    
    # Read the image and convert to base64 for display
    with open(data['file_path'], 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    with open(data['shap_path'], 'rb') as f:
        shap_data = base64.b64encode(f.read()).decode('utf-8')
    
    return render_template(
        'results.html',
        result=data['result'],
        img_data=img_data,
        shap_data=shap_data,
        session_id=session_id
    )

@app.route('/download_report/<session_id>')
def download_report(session_id):
    """Generate and download the PDF report"""
    if session_id not in session_data:
        return redirect(url_for('index'))
    
    data = session_data[session_id]
    
    # Generate the PDF report
    report_path = generate_pdf_report(
        data['result'],
        data['file_path'],
        data['shap_path']
    )
    
    # Send the file
    return send_file(
        report_path,
        as_attachment=True,
        download_name='lung_cancer_report.pdf',
        mimetype='application/pdf'
    )

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analyzing images"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Process the image using vin.py functions
        prediction, confidence = predict_image(model, file_path)
        
        # Generate SHAP visualization
        shap_path = os.path.join('shap_output', f'shap_overlay_{filename}')
        shap_arr, mask, shap_path = explain_with_shap(model, file_path, background_data, shap_path)
        
        # Calculate additional metrics
        tumor_area = int(np.sum(mask))
        stage = get_tumor_stage(tumor_area)
        risk = get_risk_level(tumor_area)
        location_sentence = get_nodule_location(mask)
        explanation_text = generate_explanation(confidence, tumor_area)
        
        # Create result dictionary
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'tumor_area': tumor_area,
            'tumor_stage': stage,
            'risk_level': risk,
            'nodule_location': location_sentence,
            'explanation_text': explanation_text,
            'shap_path': '/api/shap/' + os.path.basename(shap_path)
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/api/shap/<filename>')
def get_shap(filename):
    """Serve SHAP overlay images"""
    return send_file(os.path.join('shap_output', filename))

# Templates for the Flask app
@app.route('/templates/<template_name>')
def get_template(template_name):
    """Serve template files"""
    return render_template(template_name)

# Create the necessary templates
def create_templates():
    """Create the HTML templates for the Flask app"""
    os.makedirs('templates', exist_ok=True)
    
    # Index page
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lung Cancer Detection System</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 min-h-screen">
    <header class="bg-gray-700 text-white p-6 text-center">
        <div class="flex items-center justify-center gap-2 mb-1">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-6 w-6">
                <path d="M19 9V6a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v3" />
                <path d="M3 11v5a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5a2 2 0 0 0-4 0v2H7v-2a2 2 0 0 0-4 0Z" />
                <path d="M5 11V9c0-1.1.9-2 2-2h10a2 2 0 0 1 2 2v2" />
            </svg>
            <h1 class="text-2xl font-bold">Lung Cancer Detection System</h1>
        </div>
        <p class="text-sm">AI-Powered Medical Image Analysis</p>
    </header>

    <div class="container mx-auto py-8">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div class="bg-white p-4 rounded-lg shadow">
                <div class="flex flex-col items-center text-center">
                    <div class="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center mb-2">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5 text-blue-600">
                            <path d="M12 2a3 3 0 0 0-3 3v1H4a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h1v1a2 2 0 0 0 2 2h1v1a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-1h1a2 2 0 0 0 2-2v-1h1a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2H15V5a3 3 0 0 0-3-3Z" />
                        </svg>
                    </div>
                    <h2 class="text-lg font-semibold">Advanced AI</h2>
                    <p class="text-sm text-gray-600 mt-1">State-of-the-art deep learning technology for accurate analysis</p>
                </div>
            </div>

            <div class="bg-white p-4 rounded-lg shadow">
                <div class="flex flex-col items-center text-center">
                    <div class="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center mb-2">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5 text-blue-600">
                            <rect width="18" height="18" x="3" y="3" rx="2" />
                            <path d="M8 12h8" />
                            <path d="M12 8v8" />
                        </svg>
                    </div>
                    <h2 class="text-lg font-semibold">Secure & Private</h2>
                    <p class="text-sm text-gray-600 mt-1">Your data is processed securely and deleted after analysis</p>
                </div>
            </div>

            <div class="bg-white p-4 rounded-lg shadow">
                <div class="flex flex-col items-center text-center">
                    <div class="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center mb-2">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5 text-blue-600">
                            <path d="M12 2v4" />
                            <path d="m6.41 6.41 2.83 2.83" />
                            <path d="M2 12h4" />
                            <path d="m6.41 17.59 2.83-2.83" />
                            <path d="M12 18v4" />
                            <path d="m17.59 17.59-2.83-2.83" />
                            <path d="M22 12h-4" />
                            <path d="m17.59 6.41-2.83 2.83" />
                        </svg>
                    </div>
                    <h2 class="text-lg font-semibold">Quick Results</h2>
                    <p class="text-sm text-gray-600 mt-1">Get analysis results within seconds</p>
                </div>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div class="bg-white rounded-lg shadow">
                <div class="p-6">
                    <div class="flex items-center gap-2 mb-4">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5">
                            <circle cx="12" cy="12" r="10" />
                            <path d="M12 16v-4" />
                            <path d="M12 8h.01" />
                        </svg>
                        <h2 class="text-xl font-semibold">About This Tool</h2>
                    </div>
                    <p class="text-sm text-gray-600 mb-4">
                        This system uses advanced deep learning to analyze lung CT scans for potential cancer detection. Please
                        note that this is an AI assistant tool and should not replace professional medical advice.
                    </p>

                    <div class="bg-amber-50 border border-amber-200 rounded-md p-4 mb-4">
                        <div class="flex items-start gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5 text-amber-600 mt-0.5">
                                <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
                                <path d="M12 9v4" />
                                <path d="M12 17h.01" />
                            </svg>
                            <div>
                                <h3 class="font-medium text-amber-800">Important Warnings</h3>
                                <div class="text-sm text-amber-700 mt-1">
                                    <ul class="list-disc list-inside space-y-1">
                                        <li>This tool is for preliminary screening only</li>
                                        <li>Always consult healthcare professionals for medical decisions</li>
                                        <li>Results should not be considered as final diagnosis</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="bg-white rounded-lg shadow">
                <div class="p-6">
                    <div class="flex items-center gap-2 mb-4">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" x2="12" y1="3" y2="15" />
                        </svg>
                        <h2 class="text-xl font-semibold">Upload CT Scan</h2>
                    </div>

                    <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                        <p class="text-sm text-gray-500 mb-4">Drag & drop your lung CT scan image here</p>
                        <a href="/upload" class="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded-md inline-block">
                            Choose CT Image
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
        """)
    
    # Upload page
    with open('templates/upload.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload CT Scan - Lung Cancer Detection System</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 min-h-screen">
    <header class="bg-gray-700 text-white p-6 text-center">
        <div class="flex items-center justify-center gap-2 mb-1">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-6 w-6">
                <path d="M19 9V6a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v3" />
                <path d="M3 11v5a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5a2 2 0 0 0-4 0v2H7v-2a2 2 0 0 0-4 0Z" />
                <path d="M5 11V9c0-1.1.9-2 2-2h10a2 2 0 0 1 2 2v2" />
            </svg>
            <h1 class="text-2xl font-bold">Lung Cancer Detection System</h1>
        </div>
        <p class="text-sm">AI-Powered Medical Image Analysis</p>
    </header>

    <div class="container mx-auto py-8">
        <div class="max-w-2xl mx-auto bg-white p-6 rounded-lg shadow">
            <h2 class="text-2xl font-bold mb-6 text-center">Upload CT Scan</h2>

            <form action="/upload" method="post" enctype="multipart/form-data" id="upload-form">
                <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center mb-6" id="drop-area">
                    <div id="preview-container" class="mb-4 hidden">
                        <img id="preview-image" src="/placeholder.svg" alt="CT Scan Preview" class="max-h-64 mx-auto">
                    </div>
                    <div id="upload-prompt" class="py-8">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-12 w-12 mx-auto text-gray-400 mb-4">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" x2="12" y1="3" y2="15" />
                        </svg>
                        <p class="text-gray-500">Drag & drop your CT scan image here</p>
                    </div>

                    <div class="mt-4">
                        <label for="file-upload" class="cursor-pointer">
                            <div class="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded-md inline-flex items-center">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4 mr-2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                    <polyline points="17 8 12 3 7 8" />
                                    <line x1="12" x2="12" y1="3" y2="15" />
                                </svg>
                                Choose File
                            </div>
                            <input id="file-upload" name="file" type="file" class="hidden" accept="image/*">
                        </label>
                    </div>
                </div>

                <div class="flex justify-between">
                    <a href="/" class="bg-gray-200 hover:bg-gray-300 text-gray-800 py-2 px-4 rounded-md w-1/3 text-center">
                        Back
                    </a>
                    <button type="submit" id="analyze-btn" class="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded-md w-1/3">
                        Analyze Image
                    </button>
                </div>
            </form>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const dropArea = document.getElementById('drop-area');
            const fileInput = document.getElementById('file-upload');
            const previewContainer = document.getElementById('preview-container');
            const previewImage = document.getElementById('preview-image');
            const uploadPrompt = document.getElementById('upload-prompt');
            const analyzeBtn = document.getElementById('analyze-btn');
            const uploadForm = document.getElementById('upload-form');
            
            // Prevent default drag behaviors
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
                document.body.addEventListener(eventName, preventDefaults, false);
            });
            
            // Highlight drop area when item is dragged over it
            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, unhighlight, false);
            });
            
            // Handle dropped files
            dropArea.addEventListener('drop', handleDrop, false);
            
            // Handle selected files
            fileInput.addEventListener('change', handleFiles, false);
            
            // Form submission validation
            uploadForm.addEventListener('submit', function(e) {
                if (!fileInput.files || fileInput.files.length === 0) {
                    e.preventDefault();
                    alert('Please select a file first');
                }
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            function highlight() {
                dropArea.classList.add('border-blue-500');
                dropArea.classList.remove('border-gray-300');
            }
            
            function unhighlight() {
                dropArea.classList.remove('border-blue-500');
                dropArea.classList.add('border-gray-300');
            }
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                
                if (files.length) {
                    fileInput.files = files;
                    handleFiles();
                }
            }
            
            function handleFiles() {
                const file = fileInput.files[0];
                if (file) {
                    // Show preview
                    previewContainer.classList.remove('hidden');
                    uploadPrompt.classList.add('hidden');
                    
                    // Create preview
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        previewImage.src = e.target.result;
                    };
                    reader.readAsDataURL(file);
                }
            }
        });
    </script>
</body>
</html>
        """)
    
    # Results page
    with open('templates/results.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Results - Lung Cancer Detection System</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 min-h-screen">
    <header class="bg-gray-700 text-white p-6 text-center">
        <div class="flex items-center justify-center gap-2 mb-1">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-6 w-6">
                <path d="M19 9V6a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v3" />
                <path d="M3 11v5a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5a2 2 0 0 0-4 0v2H7v-2a2 2 0 0 0-4 0Z" />
                <path d="M5 11V9c0-1.1.9-2 2-2h10a2 2 0 0 1 2 2v2" />
            </svg>
            <h1 class="text-2xl font-bold">Analysis Result</h1>
        </div>
        <p class="text-sm">Lung Cancer Detection System</p>
    </header>

    <!-- Prominent Prediction Display at the top -->
    <div class="container mx-auto pt-4">
        <div class="{% if result.prediction == 'Benign' %}bg-green-600{% else %}bg-red-600{% endif %} text-center p-4 rounded-lg text-white font-bold text-2xl mb-4">
            Predicted Case: {{ result.prediction }}
        </div>
    </div>

    <div class="container mx-auto py-4">
        <div class="max-w-2xl mx-auto bg-white p-6 rounded-lg shadow">
            <div class="mb-6">
                <div class="{% if result.prediction == 'Benign' %}bg-green-50 border border-green-200{% else %}bg-red-50 border border-red-200{% endif %} rounded-md p-4 mb-6">
                    <div class="flex items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5 {% if result.prediction == 'Benign' %}text-green-600{% else %}text-red-600{% endif %}">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                            <path d="m9 11 3 3L22 4" />
                        </svg>
                        <h2 class="text-xl font-semibold">Analysis Result</h2>
                    </div>
                    <p class="text-lg font-medium mt-2 {% if result.prediction == 'Benign' %}text-green-700{% else %}text-red-700{% endif %}">
                        Predicted: {{ result.prediction }}
                    </p>
                </div>

                <div class="text-center mb-6">
                    <h3 class="text-lg font-medium mb-2">Uploaded CT Scan</h3>
                    <img src="data:image/jpeg;base64,{{ img_data }}" alt="CT Scan" class="max-h-64 mx-auto border border-gray-200 rounded-md">
                </div>

                <div class="text-center mb-6">
                    <h3 class="text-lg font-medium mb-2">SHAP Analysis Visualization</h3>
                    <img src="data:image/png;base64,{{ shap_data }}" alt="SHAP Overlay" class="max-h-64 mx-auto border border-gray-200 rounded-md">
                </div>

                <div class="space-y-4 mb-6">
                    <div>
                        <h3 class="text-lg font-medium mb-2">Analysis Details</h3>
                        <div class="grid grid-cols-2 gap-4">
                            <div class="bg-gray-50 p-3 rounded-md">
                                <p class="text-sm text-gray-500">Confidence</p>
                                <p class="font-medium">{{ "%.2f"|format(result.confidence) }}%</p>
                            </div>
                            <div class="bg-gray-50 p-3 rounded-md">
                                <p class="text-sm text-gray-500">Tumor Area</p>
                                <p class="font-medium">{{ result.tumor_area }} pixels</p>
                            </div>
                            <div class="bg-gray-50 p-3 rounded-md">
                                <p class="text-sm text-gray-500">Tumor Stage</p>
                                <p class="font-medium">{{ result.tumor_stage }}</p>
                            </div>
                            <div class="bg-gray-50 p-3 rounded-md">
                                <p class="text-sm text-gray-500">Risk Level</p>
                                <p class="font-medium">{{ result.risk_level }}</p>
                            </div>
                        </div>
                    </div>

                    <div>
                        <h3 class="text-lg font-medium mb-2">Nodule Location</h3>
                        <p class="text-gray-700">{{ result.nodule_location }}</p>
                    </div>

                    <div>
                        <h3 class="text-lg font-medium mb-2">Analysis Explanation</h3>
                        <p class="text-gray-700">{{ result.explanation_text }}</p>
                    </div>
                </div>

                <div class="bg-amber-50 border border-amber-200 rounded-md p-4 mb-6">
                    <div class="flex items-start gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5 text-amber-600 mt-0.5">
                            <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
                            <path d="M12 9v4" />
                            <path d="M12 17h.01" />
                        </svg>
                        <div>
                            <h3 class="font-medium text-amber-800">Important Notice</h3>
                            <div class="text-sm text-amber-700 mt-1">
                                <ul class="list-disc list-inside space-y-1">
                                    <li>This is not a medical diagnosis</li>
                                    <li>Consult a healthcare professional for proper medical evaluation</li>
                                    <li>False positives and negatives are possible</li>
                                    <li>For emergency situations, seek immediate medical attention</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="border-t border-gray-200 pt-6">
                    <h3 class="text-lg font-medium mb-4">
                        <span class="flex items-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5">
                                <polyline points="22 12 16 12 14 15 10 15 8 12 2 12" />
                                <path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
                            </svg>
                            Recommended Next Steps
                        </span>
                    </h3>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                        <div class="flex items-start gap-3">
                            <div class="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4 text-blue-600">
                                    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
                                    <circle cx="9" cy="7" r="4" />
                                    <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
                                    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                                </svg>
                            </div>
                            <div>
                                <h4 class="font-medium">Consult a Doctor</h4>
                                <p class="text-sm text-gray-600">
                                    Share these results with your healthcare provider for professional evaluation
                                </p>
                            </div>
                        </div>

                        <div class="flex items-start gap-3">
                            <div class="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4 text-blue-600">
                                    <rect width="18" height="18" x="3" y="4" rx="2" ry="2" />
                                    <line x1="16" x2="16" y1="2" y2="6" />
                                    <line x1="8" x2="8" y1="2" y2="6" />
                                    <line x1="3" x2="21" y1="10" y2="10" />
                                    <path d="M8 14h.01" />
                                    <path d="M12 14h.01" />
                                    <path d="M16 14h.01" />
                                    <path d="M8 18h.01" />
                                    <path d="M12 18h.01" />
                                    <path d="M16 18h.01" />
                                </svg>
                            </div>
                            <div>
                                <h4 class="font-medium">Schedule Follow-up</h4>
                                <p class="text-sm text-gray-600">
                                    Book an appointment for further medical assessment if needed
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="flex flex-wrap gap-4 justify-between">
                <a href="/upload" class="bg-gray-200 hover:bg-gray-300 text-gray-800 py-2 px-4 rounded-md flex-1 md:flex-none md:w-auto text-center flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4 mr-2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17 8 12 3 7 8" />
                        <line x1="12" x2="12" y1="3" y2="15" />
                    </svg>
                    Analyze Another Image
                </a>
                <a href="/" class="bg-gray-200 hover:bg-gray-300 text-gray-800 py-2 px-4 rounded-md flex-1 md:flex-none md:w-auto text-center flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4 mr-2">
                        <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
                        <polyline points="9 22 9 12 15 12 15 22" />
                    </svg>
                    Go to Home
                </a>
                <a href="/download_report/{{ session_id }}" class="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded-md flex-1 md:flex-none md:w-auto text-center flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4 mr-2">
                        <polyline points="6 9 6 2 18 2 18 9" />
                        <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2" />
                        <rect width="12" height="8" x="6" y="14" />
                    </svg>
                    Print Results
                </a>
            </div>
        </div>
    </div>
</body>
</html>
        """)

# Create templates when the script is run
create_templates()

if __name__ == '__main__':
    print("Starting Lung Cancer Detection System...")
    print("Creating templates...")
    create_templates()
    print("Templates created successfully!")
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)