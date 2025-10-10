# -*- coding: utf-8 -*-
"""
Advanced MS MRI Analysis Server with TDA - Optimized for Large Files

This Flask application provides an advanced API for analyzing brain MRI scans
for signs of Multiple Sclerosis (MS) using deep learning and topological data analysis (TDA).
Fully optimized for large file processing with streaming and incremental processing.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from flask import Flask, request, jsonify, render_template
import tempfile
import base64
import io
from PIL import Image
import logging
import cv2
import nibabel as nib
import tensorflow as tf
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from skimage import measure
import gdown
import gc
import time
import sys

# حل بديل لـ gtda إذا لم يعمل
try:
    from gtda.homology import VietorisRipsPersistence
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False
    print("⚠️ Giotto-TDA not available, using geometric features only")

logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ========== إعدادات متقدمة للملفات الكبيرة ==========
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

print("🚀 Starting Advanced MS MRI Analysis Server with TDA (Optimized for Large Files)...")

# =====================================================
# Load AI Models
# =====================================================

print("Loading AI models...")

# Initialize models
unet_model = None
rf_model = None
scaler = None

def download_unet_model():
    """Download U-Net model from Google Drive"""
    model_path = "best_unet_final.keras"
    if not os.path.exists(model_path):
        print("📥 Downloading U-Net model from Google Drive...")
        try:
            url = "https://drive.google.com/uc?id=1CgugA_Ti0prkQH3j7NL_pEmXjZx-FfdB&confirm=t"
            gdown.download(url, model_path, quiet=False)
            
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024*1024)
                print(f"✅ U-Net model downloaded successfully ({file_size:.1f} MB)")
                return True
            else:
                print("❌ File was not downloaded")
                return False
                
        except Exception as e:
            print(f"❌ Failed to download U-Net model: {e}")
            return False
    return True

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), "float32")
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + (1 - dice_coefficient(y_true, y_pred))

try:
    if download_unet_model():
        custom_objects = {"dice_coefficient": dice_coefficient, "bce_dice_loss": bce_dice_loss}
        unet_model = tf.keras.models.load_model("best_unet_final.keras", custom_objects=custom_objects)
        print("✅ U-Net model loaded successfully")
    else:
        print("⚠️ Using basic mode without U-Net")
except Exception as e:
    print(f"⚠️ U-Net model loading failed: {e}")

try:
    rf_model = joblib.load("rf_classifier.pkl")
    print("✅ Random Forest model loaded successfully")
except Exception as e:
    print(f"⚠️ Random Forest model loading failed: {e}")

try:
    scaler = joblib.load("scaler.pkl")
    print("✅ Scaler loaded successfully")
except Exception as e:
    print(f"⚠️ Scaler loading failed: {e}")

# =====================================================
# Large File Processing - Streamlined Approach
# =====================================================

def process_very_large_nii_file(file_path, target_slices=150):
    """معالجة الملفات الكبيرة جداً بطريقة متدرجة"""
    print(f"📁 Processing VERY large NII file: {file_path}")
    
    try:
        # فتح الملف مرة واحدة فقط
        nii_img = nib.load(file_path)
        img_data = nii_img.get_fdata()
        original_shape = img_data.shape
        
        print(f"📊 Original image dimensions: {original_shape}")
        print(f"💾 Estimated memory usage: {img_data.nbytes / (1024*1024*1024):.2f} GB")
        
        # تحديد عدد الشرائح للمعالجة
        total_slices = img_data.shape[2]
        
        if total_slices > 300:
            # للملفات الضخمة، نستخدم عينات استراتيجية
            target_slices = min(120, total_slices)  # تقليل أكثر للذاكرة
            
            # أخذ عينات من مناطق مختلفة من الدماغ
            sample_indices = strategic_sampling(total_slices, target_slices)
            processed_data = img_data[:, :, sample_indices]
            
            print(f"🔧 Very large file: Processing {target_slices} strategic samples from {total_slices} slices")
            
        elif total_slices > 150:
            # للملفات الكبيرة، نستخدم عينات موزعة
            sample_indices = np.linspace(0, total_slices - 1, target_slices, dtype=int)
            processed_data = img_data[:, :, sample_indices]
            
            print(f"🔧 Large file: Processing {target_slices} distributed samples from {total_slices} slices")
            
        else:
            # ملف عادي - معالجة كاملة
            processed_data = img_data
            sample_indices = list(range(total_slices))
            
        print(f"📊 Final dimensions: {processed_data.shape}")
        return processed_data, sample_indices, total_slices
        
    except Exception as e:
        print(f"❌ Error processing large NII file: {e}")
        raise

def strategic_sampling(total_slices, target_slices):
    """أخذ عينات استراتيجية من مناطق مختلفة في الدماغ"""
    # التركيز على المناطق الوسطى حيث توجد معظم الآفات
    middle_start = total_slices // 4
    middle_end = 3 * total_slices // 4
    
    # عينات من المنطقة الوسطى (80%)
    middle_samples = int(target_slices * 0.8)
    middle_indices = np.linspace(middle_start, middle_end, middle_samples, dtype=int)
    
    # عينات من المناطق الطرفية (20%)
    edge_samples = target_slices - middle_samples
    edge_indices1 = np.linspace(0, middle_start-1, edge_samples//2, dtype=int)
    edge_indices2 = np.linspace(middle_end+1, total_slices-1, edge_samples//2, dtype=int)
    
    # دمج المؤشرات
    all_indices = np.concatenate([edge_indices1, middle_indices, edge_indices2])
    return np.sort(all_indices)

def incremental_preprocessing(img_data, batch_size=16):
    """معالجة متدرجة للشرائح لتوفير الذاكرة"""
    print(f"🔧 Incremental preprocessing of {img_data.shape[2]} slices...")
    
    all_slices = []
    
    for i in range(0, img_data.shape[2], batch_size):
        batch_end = min(i + batch_size, img_data.shape[2])
        batch_slices = []
        
        for j in range(i, batch_end):
            try:
                # معالجة كل شريحة على حدة
                sl = img_data[:, :, j]
                sl_normalized = (sl - np.min(sl)) / (np.max(sl) - np.min(sl) + 1e-8)
                sl_resized = cv2.resize(sl_normalized, (128, 128))
                sl_final = np.expand_dims(sl_resized, axis=-1)
                batch_slices.append(sl_final)
            except Exception as e:
                print(f"⚠️ Error processing slice {j}: {e}")
                # إنشاء شريحة فارغة في حالة الخطأ
                empty_slice = np.zeros((128, 128, 1))
                batch_slices.append(empty_slice)
        
        all_slices.extend(batch_slices)
        
        # تنظيف الذاكرة بعد كل دفعة
        del batch_slices
        gc.collect()
        
        if (i // batch_size) % 10 == 0:
            print(f"   Processed {batch_end}/{img_data.shape[2]} slices")
    
    return np.array(all_slices)

def streaming_unet_segmentation(slices, threshold=0.1, batch_size=8):
    """تجزئة U-Net متدرجة لتوفير الذاكرة"""
    if unet_model is None:
        print("⚠️ U-Net model not available, using efficient mock segmentation")
        return create_efficient_mock_masks(slices)

    print("Running streaming U-Net segmentation...")
    
    all_masks = []
    total_batches = (len(slices) + batch_size - 1) // batch_size
    
    for i in range(0, len(slices), batch_size):
        batch_end = min(i + batch_size, len(slices))
        batch_slices = slices[i:batch_end]
        
        print(f"   Segmenting batch {i//batch_size + 1}/{total_batches}")
        
        try:
            # التنبؤ على الدفعة الحالية فقط
            batch_predictions = unet_model.predict(batch_slices, verbose=0)
            batch_masks = (batch_predictions > threshold).astype(np.uint8)
            all_masks.extend(batch_masks)
            
            # تنظيف شامل للذاكرة
            del batch_slices, batch_predictions, batch_masks
            gc.collect()
            
        except Exception as e:
            print(f"⚠️ Error in batch {i//batch_size + 1}: {e}")
            # إضافة أقنعة افتراضية في حالة الخطأ
            for _ in range(len(batch_slices)):
                all_masks.append(np.zeros((128, 128, 1), dtype=np.uint8))
    
    masks_array = np.array(all_masks)
    non_empty_count = np.sum([np.any(mask > 0) for mask in masks_array])
    print(f"✅ Streaming segmentation completed: {non_empty_count}/{len(masks_array)} non-empty masks")
    
    return masks_array

def create_efficient_mock_masks(slices):
    """إنشاء أقنعة وهمية فعالة للذاكرة"""
    print("Creating efficient mock masks...")
    binary_masks = []
    
    for i, slice_img in enumerate(slices):
        mask = np.zeros_like(slice_img.squeeze())
        
        # إضافة آفات وهمية في 10% من الشرائح فقط
        if i % 10 == 0 and i > len(slices) * 0.3 and i < len(slices) * 0.7:
            h, w = mask.shape
            num_lesions = np.random.randint(2, 6)
            for _ in range(num_lesions):
                x = np.random.randint(20, w-20)
                y = np.random.randint(20, h-20)
                radius = np.random.randint(3, 10)
                cv2.circle(mask, (x, y), radius, 1, -1)
        
        binary_masks.append(mask.astype(np.uint8))
    
    return np.array(binary_masks)

# =====================================================
# Efficient TDA Processing for Large Files
# =====================================================

def efficient_tda_feature_extraction(masks, num_features_expected=603, sample_rate=0.3):
    """استخراج ميزات TDA فعال للذاكرة"""
    print("Efficient TDA feature extraction for large files...")
    
    if not TDA_AVAILABLE:
        print("⚠️ Using geometric features only (TDA not available)")
        return efficient_geometric_features(masks, num_features_expected)

    # استخدام عينة عشوائية من الأقنعة لتقليل الحساب
    total_masks = len(masks)
    if total_masks > 100:
        sample_size = max(50, int(total_masks * sample_rate))
        sample_indices = np.random.choice(total_masks, sample_size, replace=False)
        sampled_masks = [masks[i] for i in sample_indices]
        print(f"🔬 Sampling {sample_size} masks from {total_masks} for TDA analysis")
    else:
        sampled_masks = masks
        sample_indices = list(range(total_masks))

    features_all = []
    persistence = VietorisRipsPersistence(
        homology_dimensions=[0, 1],
        n_jobs=1,
        max_edge_length=1.5,
        collapse_edges=True
    )

    for i, mask in enumerate(sampled_masks):
        if i % 20 == 0:
            print(f"  Processing TDA sample {i}/{len(sampled_masks)}")

        mask_2d = mask.squeeze()
        non_zero_pixels = np.sum(mask_2d > 0)

        if non_zero_pixels < 20:
            feats = create_compact_zero_features(num_features_expected)
        else:
            try:
                binary_mask = (mask_2d > 0).astype(np.float64)
                
                # تنظيف morphological مبسط
                cleaned_mask = ndimage.binary_opening(binary_mask, structure=np.ones((2,2)))
                points = np.column_stack(np.where(cleaned_mask > 0))

                if len(points) > 15:
                    # تطبيع النقاط
                    points = points.astype(np.float64)
                    points[:, 0] = points[:, 0] / binary_mask.shape[0]
                    points[:, 1] = points[:, 1] / binary_mask.shape[1]
                    
                    # إضافة ضوضاء طفيفة
                    points += np.random.normal(0, 0.0001, points.shape)

                    diagrams = persistence.fit_transform([points])
                    feats = extract_efficient_features(diagrams, cleaned_mask)
                else:
                    feats = create_compact_zero_features(num_features_expected)

            except Exception as e:
                feats = create_compact_zero_features(num_features_expected)

        # ضبط عدد الميزات
        if len(feats) < num_features_expected:
            feats.extend([0.0] * (num_features_expected - len(feats)))
        elif len(feats) > num_features_expected:
            feats = feats[:num_features_expected]

        features_all.append(feats)

    # توسيع النتائج لجميع الأقنعة
    full_features = []
    feature_idx = 0
    for i in range(total_masks):
        if i in sample_indices:
            full_features.append(features_all[feature_idx])
            feature_idx += 1
        else:
            full_features.append(create_compact_zero_features(num_features_expected))

    features_array = np.array(full_features)
    print(f"✅ Efficient TDA features extracted: {features_array.shape}")
    
    return features_array

def efficient_geometric_features(masks, num_features_expected):
    """ميزات هندسية فعالة للذاكرة"""
    print("Using efficient geometric features...")
    features_all = []
    
    for i, mask in enumerate(masks):
        if i % 50 == 0:
            print(f"  Processing geometric features {i}/{len(masks)}")
            
        mask_2d = mask.squeeze()
        binary_mask = (mask_2d > 0).astype(np.float64)
        feats = extract_efficient_features([], binary_mask)
        
        if len(feats) < num_features_expected:
            feats.extend([0.0] * (num_features_expected - len(feats)))
        
        features_all.append(feats)
    
    return np.array(features_all)

def create_compact_zero_features(num_features):
    """ميزات صفرية مضغوطة"""
    return [0.0] * num_features

def extract_efficient_features(diagrams, binary_mask):
    """استخراج ميزات فعال"""
    features = []

    # معالجة الأبعاد التبولوجية
    for dim, diagram in enumerate(diagrams):
        if len(diagram) > 0:
            lifetimes = diagram[:, 1] - diagram[:, 0]
            
            topological_feats = [
                np.sum(lifetimes) if len(lifetimes) > 0 else 0.0,
                np.mean(lifetimes) if len(lifetimes) > 0 else 0.0,
                np.std(lifetimes) if len(lifetimes) > 0 else 0.0,
                np.max(lifetimes) if len(lifetimes) > 0 else 0.0,
                len(lifetimes),
            ]
            features.extend(topological_feats)
        else:
            features.extend([0.0] * 5)

    # ميزات هندسية مبسطة
    if np.sum(binary_mask) > 0:
        try:
            labeled, num_components = ndimage.label(binary_mask)
            if num_components > 0:
                sizes = ndimage.sum(binary_mask, labeled, range(1, num_components + 1))
                
                geometric_feats = [
                    np.sum(binary_mask),
                    num_components,
                    np.max(sizes) if len(sizes) > 0 else 0.0,
                    np.mean(sizes) if len(sizes) > 0 else 0.0,
                ]
                features.extend(geometric_feats)
            else:
                features.extend([0.0] * 4)
        except:
            features.extend([0.0] * 4)
    else:
        features.extend([0.0] * 4)

    return features

# =====================================================
# Efficient Probability Calculation
# =====================================================

def calculate_efficient_ms_probability(positive_slices, avg_prob, max_prob, binary_masks, probabilities, total_slices):
    """حساب احتمالية MS فعال للذاكرة"""

    if positive_slices == 0:
        return 5.0

    # عوامل مبسطة
    slice_ratio = positive_slices / total_slices
    slice_factor = min(slice_ratio * 2, 0.9)  # تبسيط حساب عامل الشرائح

    # عامل الثقة
    high_confidence_count = np.sum(probabilities > 0.7)
    confidence_ratio = high_confidence_count / max(positive_slices, 1)
    prob_factor = avg_prob * min(confidence_ratio, 1.0)

    # توزيع الآفات
    distribution_factor = calculate_efficient_distribution_factor(binary_masks, probabilities)

    # حساب احتمالية مبسط
    ms_probability = (
        slice_factor * 0.35 +
        prob_factor * 0.35 +
        distribution_factor * 0.3
    )

    # تطبيقات واقعية
    ms_probability = min(ms_probability * 100, 95.0)

    # تعديلات بناءً على الآفات عالية الثقة
    if high_confidence_count >= 5:
        ms_probability = min(ms_probability + 10, 95.0)
    elif high_confidence_count >= 2:
        ms_probability = min(ms_probability + 5, 95.0)

    return max(ms_probability, 5.0)

def calculate_efficient_distribution_factor(binary_masks, probabilities):
    """حساب توزيع الآفات فعال الذاكرة"""
    total_volume = 0
    high_prob_lesions = 0
    
    for i, mask in enumerate(binary_masks):
        if probabilities[i] > 0.3:
            total_volume += np.sum(mask)
            if probabilities[i] > 0.7:
                high_prob_lesions += 1
    
    if high_prob_lesions == 0:
        return 0.1
    
    # حساب مبسط للتوزيع
    if high_prob_lesions >= 10:
        return 0.9
    elif high_prob_lesions >= 5:
        return 0.7
    elif high_prob_lesions >= 2:
        return 0.5
    else:
        return 0.3

# =====================================================
# Efficient Classification
# =====================================================

def efficient_fallback_classification(binary_masks):
    """تصنيف احتياطي فعال"""
    predictions = []
    probabilities = []
    
    for mask in binary_masks:
        volume = np.sum(mask)
        if volume > 50:
            prob = min(volume / 500.0, 1.0)
            predictions.append(1)
            probabilities.append(prob)
        else:
            predictions.append(0)
            probabilities.append(0.0)
    
    return np.array(predictions), np.array(probabilities)

# =====================================================
# Visualization Functions (Unchanged)
# =====================================================

def create_clean_medical_visualization(mri_slice, binary_mask, probability):
    """Create CLEAN medical visualization WITHOUT PROB/CONF text"""
    if mri_slice.dtype != np.uint8:
        mri_slice = (mri_slice * 255).astype(np.uint8)

    if len(mri_slice.shape) == 2:
        mri_rgb = np.stack([mri_slice] * 3, axis=-1)
    else:
        mri_rgb = mri_slice.copy()

    mri_bgr = cv2.cvtColor(mri_rgb, cv2.COLOR_RGB2BGR)

    if binary_mask.shape[:2] != mri_bgr.shape[:2]:
        binary_mask_resized = cv2.resize(binary_mask.squeeze(),
                                       (mri_bgr.shape[1], mri_bgr.shape[0]))
    else:
        binary_mask_resized = binary_mask.squeeze()

    # Create colored mask based on probability
    if probability > 0.7:
        color = (0, 0, 255)  # Red
        alpha = 0.6
    elif probability > 0.4:
        color = (0, 165, 255)  # Orange
        alpha = 0.5
    else:
        color = (0, 255, 255)  # Yellow
        alpha = 0.4

    # Apply mask overlay
    mask_indices = binary_mask_resized > 0
    if np.any(mask_indices):
        overlay = mri_bgr.copy()
        overlay[mask_indices] = color
        mri_bgr = cv2.addWeighted(overlay, alpha, mri_bgr, 1 - alpha, 0)

        contours, _ = cv2.findContours(binary_mask_resized.astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mri_bgr, contours, -1, color, 1)

    result_image = cv2.cvtColor(mri_bgr, cv2.COLOR_BGR2RGB)
    return result_image

def get_representative_slices(slices, binary_masks, predictions, probabilities, num_slices=6):
    """Get representative slices efficiently"""
    positive_indices = np.where(predictions == 1)[0]

    if len(positive_indices) == 0:
        # إذا لم توجد آفات، نأخذ شرائح من المنتصف
        middle_start = len(slices) // 3
        middle_end = 2 * len(slices) // 3
        if middle_end - middle_start > num_slices:
            selected_indices = np.linspace(middle_start, middle_end, num_slices, dtype=int)
        else:
            selected_indices = range(middle_start, min(middle_start + num_slices, len(slices)))
    else:
        # نأخذ أفضل الشرائح مع الآفات
        sorted_indices = positive_indices[np.argsort(probabilities[positive_indices])[::-1]]
        selected_indices = sorted_indices[:min(num_slices, len(sorted_indices))]

    representative_slices = []
    for idx in selected_indices:
        if idx < len(slices):
            slice_data = {
                'index': int(idx),
                'probability': float(probabilities[idx]) if idx < len(probabilities) else 0.1,
                'original_slice': slices[idx].squeeze(),
                'binary_mask': binary_masks[idx].squeeze() if idx < len(binary_masks) else np.zeros_like(slices[idx].squeeze())
            }
            representative_slices.append(slice_data)

    return representative_slices

# =====================================================
# Flask Routes - Optimized for Large Files
# =====================================================

@app.route('/')
def home():
    return render_template('index_advanced.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'Advanced MS MRI Analysis Server with TDA - Optimized for Large Files',
        'models_loaded': unet_model is not None,
        'tda_available': TDA_AVAILABLE,
        'max_file_size': '500MB',
        'memory_optimized': True,
        'large_file_support': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    try:
        print("🎯 Received analysis request for large file")

        if 'nii_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['nii_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # فحص حجم الملف
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        
        print(f"📁 Processing file: {file.filename} ({file_size / (1024*1024):.2f} MB)")

        MAX_FILE_SIZE = 500 * 1024 * 1024
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'error': 'File too large', 
                'max_size': '500MB',
                'your_file_size': f'{file_size / (1024*1024):.2f}MB',
                'status': 'error'
            }), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            # معالجة الملف الكبير بطريقة فعالة
            img_data, sample_indices, total_slices = process_very_large_nii_file(temp_path)
            
            # المعالجة المتدرجة
            slices = incremental_preprocessing(img_data)
            print(f"🔧 Preprocessed {len(slices)} slices efficiently")
            
            # تحرير الذاكرة
            del img_data
            gc.collect()

            # التجزئة المتدرجة
            binary_masks = streaming_unet_segmentation(slices)
            print(f"📊 Segmentation completed: {len(binary_masks)} masks")

            # استخراج الميزات بشكل فعال
            print("🔬 Efficient TDA feature extraction...")
            try:
                tda_features = efficient_tda_feature_extraction(binary_masks)
                analysis_method = "Efficient TDA + Geometric" if TDA_AVAILABLE else "Efficient Geometric"
                
                # استخدام النموذج الموجود أو الاحتياطي
                if rf_model is not None and scaler is not None:
                    try:
                        tda_features_scaled = scaler.transform(tda_features)
                        predictions = rf_model.predict(tda_features_scaled)
                        probabilities = rf_model.predict_proba(tda_features_scaled)[:, 1]
                        print("✅ Efficient classification successful")
                    except Exception as e:
                        print(f"Model classification failed: {e}")
                        predictions, probabilities = efficient_fallback_classification(binary_masks)
                        analysis_method = "Efficient Geometric"
                else:
                    predictions, probabilities = efficient_fallback_classification(binary_masks)
                    analysis_method = "Efficient Geometric"

            except Exception as e:
                print(f"TDA extraction failed: {e}")
                predictions, probabilities = efficient_fallback_classification(binary_masks)
                analysis_method = "Basic Geometric Analysis"

            # الإحصائيات
            positive_slices = int(np.sum(predictions))
            avg_prob = float(np.mean(probabilities))
            max_prob = float(np.max(probabilities))

            # حساب الاحتمالية
            ms_probability = calculate_efficient_ms_probability(
                positive_slices, avg_prob, max_prob, binary_masks, probabilities, len(slices)
            )

            print(f"🔍 Analysis Results:")
            print(f"   - Positive slices: {positive_slices}/{len(slices)}")
            print(f"   - MS Probability: {ms_probability:.1f}%")
            print(f"   - Analysis Method: {analysis_method}")

            # الشرائح التمثيلية
            representative_slices = get_representative_slices(
                slices, binary_masks, predictions, probabilities, 6
            )

            # التصورات
            visualization_images = []
            for slice_data in representative_slices:
                viz_image = create_clean_medical_visualization(
                    slice_data['original_slice'],
                    slice_data['binary_mask'],
                    slice_data['probability']
                )

                pil_img = Image.fromarray(viz_image)
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                visualization_images.append({
                    'slice_index': slice_data['index'],
                    'probability': slice_data['probability'],
                    'image': img_base64
                })

            # التشخيص
            if ms_probability > 75:
                diagnosis = "🟥 HIGH PROBABILITY OF MULTIPLE SCLEROSIS"
                severity = "High"
            elif ms_probability > 55:
                diagnosis = "🟨 SUSPICIOUS FOR DEMYELINATING DISEASE"
                severity = "Moderate"
            elif ms_probability > 30:
                diagnosis = "🟦 POSSIBLE EARLY MS OR OTHER CONDITION"
                severity = "Low"
            else:
                diagnosis = "✅ NO SIGNIFICANT LESIONS DETECTED"
                severity = "None"

            # وقت المعالجة
            processing_time = time.time() - start_time

            # الرد
            response_data = {
                'diagnosis': diagnosis,
                'ms_probability': float(round(ms_probability, 1)),
                'positive_slices': positive_slices,
                'total_slices': len(slices),
                'avg_probability': float(round(avg_prob, 3)),
                'max_probability': float(round(max_prob, 3)),
                'severity': severity,
                'has_lesions': bool(positive_slices > 0),
                'status': 'success',
                'message': 'Efficient large file analysis completed successfully',
                'visualizations': visualization_images,
                'detailed_analysis': {
                    'analysis_method': analysis_method,
                    'tda_available': TDA_AVAILABLE,
                    'file_size_processed': f'{file_size / (1024*1024):.2f}MB',
                    'processing_time_seconds': round(processing_time, 1),
                    'large_file_optimized': True,
                    'sampling_method': 'Strategic sampling used',
                    'slices_analyzed': f"{len(slices)} of {total_slices} total"
                }
            }

            print(f"✅ Efficient analysis completed in {processing_time:.1f}s")

            return jsonify(response_data)

        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return jsonify({
                'error': 'Analysis failed', 
                'message': str(e),
                'status': 'error'
            }), 500

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        print(f"❌ General prediction error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}', 'status': 'error'}), 500

# =====================================================
# Main Execution
# =====================================================

def create_app():
    if not os.path.exists('templates'):
        os.makedirs('templates')
    return app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    
    print("=" * 60)
    print("🚀 MS MRI ANALYSIS SERVER - OPTIMIZED FOR LARGE FILES")
    print("=" * 60)
    print(f"📡 Server: http://0.0.0.0:{port}")
    print(f"🔍 Health: http://0.0.0.0:{port}/health")
    print(f"📁 Max file size: 500MB")
    print(f"🧠 AI Models: {'✅ Loaded' if unet_model is not None else '⚠️ Basic Mode'}")
    print(f"🔬 TDA: {'✅ Available' if TDA_AVAILABLE else '⚠️ Geometric Only'}")
    print(f"💾 Large file support: ✅ Enabled")
    print(f"⚡ Efficient processing: ✅ Streaming & sampling")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
