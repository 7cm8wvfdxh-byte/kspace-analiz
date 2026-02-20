"""
K-Space Analysis Engine
========================
Core analysis module for the web application.
Refactored from kspace_research.py for modular use.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.ndimage import zoom
import pydicom
import os
import json
import uuid
import csv
from datetime import datetime

# Import our new AI K-Space module
import sys
import os
sys.path.append(os.path.dirname(__file__))
from ai_detector import generate_kspace_fingerprint, image_free_pathology_detection


def load_dicom_series(path):
    """Load DICOM series from directory or DICOMDIR."""
    images = []
    metadata = {}

    if os.path.isdir(path):
        records = []
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath):
                try:
                    dcm = pydicom.dcmread(fpath)
                    inst = int(dcm.get('InstanceNumber', 0))
                    records.append((inst, dcm.pixel_array.astype(float), dcm))
                except:
                    pass
        records.sort(key=lambda x: x[0])
        images = [img for _, img, _ in records]

        if records:
            dcm = records[0][2]
            metadata = {
                'patient_name': str(dcm.get('PatientName', 'Unknown')),
                'patient_id': str(dcm.get('PatientID', '')),
                'patient_age': str(dcm.get('PatientAge', '')),
                'patient_sex': str(dcm.get('PatientSex', '')),
                'study_date': str(dcm.get('StudyDate', '')),
                'series_description': str(dcm.get('SeriesDescription', '')),
                'modality': str(dcm.get('Modality', '')),
                'body_part': str(dcm.get('BodyPartExamined', '')),
                'rows': int(dcm.get('Rows', 0)),
                'columns': int(dcm.get('Columns', 0)),
                'slice_count': len(records),
                'institution': str(dcm.get('InstitutionName', '')),
            }
    elif os.path.basename(path) == "DICOMDIR":
        ds = pydicom.dcmread(path)
        base_dir = os.path.dirname(path) or '.'
        records = []
        for record in ds.DirectoryRecordSequence:
            if record.DirectoryRecordType == "IMAGE":
                parts = record.ReferencedFileID
                if isinstance(parts, str):
                    parts = [parts]
                image_path = os.path.join(base_dir, *parts)
                try:
                    dcm = pydicom.dcmread(image_path)
                    inst = int(dcm.get('InstanceNumber', 0))
                    records.append((inst, dcm.pixel_array.astype(float), dcm))
                except:
                    pass
        records.sort(key=lambda x: x[0])
        images = [img for _, img, _ in records]
        if records:
            dcm = records[0][2]
            metadata = {
                'patient_name': str(dcm.get('PatientName', 'Unknown')),
                'patient_id': str(dcm.get('PatientID', '')),
                'series_description': str(dcm.get('SeriesDescription', '')),
                'modality': str(dcm.get('Modality', '')),
                'slice_count': len(records),
            }
    else:
        dcm = pydicom.dcmread(path)
        images = [dcm.pixel_array.astype(float)]
        metadata = {
            'series_description': str(dcm.get('SeriesDescription', '')),
            'modality': str(dcm.get('Modality', '')),
            'slice_count': 1,
        }

    return images, metadata


def compute_kspace(image):
    return np.fft.fftshift(np.fft.fft2(image))


# ==========================================================================
# MODULE 1: Differential Analysis
# ==========================================================================

def differential_analysis(images):
    """Compute slice-to-slice K-space differences."""
    if len(images) < 2:
        return {'delta_magnitudes': [], 'anomaly_scores': [], 'anomaly_maps': [],
                'cumulative_anomaly': None, 'transitions': []}

    kspaces = [compute_kspace(img) for img in images]
    delta_magnitudes = []
    delta_phases = []

    for i in range(len(kspaces) - 1):
        dk = kspaces[i + 1] - kspaces[i]
        delta_magnitudes.append(np.abs(dk))
        delta_phases.append(np.angle(dk))

    dk_stack = np.array(delta_magnitudes)
    global_mean = np.mean(dk_stack, axis=0)
    global_std = np.std(dk_stack, axis=0) + 1e-10

    anomaly_maps = []
    anomaly_scores = []
    transitions = []

    threshold = np.mean(anomaly_scores) + 2 * np.std(anomaly_scores) if anomaly_scores else 999

    for i, dk_mag in enumerate(delta_magnitudes):
        z_score = (dk_mag - global_mean) / global_std
        anomaly_score = float(np.mean(np.abs(z_score)))
        anomaly_scores.append(anomaly_score)
        anomaly_maps.append(z_score)

    threshold = np.mean(anomaly_scores) + 2 * np.std(anomaly_scores)

    for i, score in enumerate(anomaly_scores):
        transitions.append({
            'from_slice': i + 1,
            'to_slice': i + 2,
            'dk_score': round(score, 4),
            'status': 'ANOMALY' if score > threshold else 'Normal',
            'is_anomaly': score > threshold,
        })

    cumulative_anomaly = np.max(np.abs(np.stack(anomaly_maps, axis=0)), axis=0)

    return {
        'delta_magnitudes': delta_magnitudes,
        'anomaly_scores': anomaly_scores,
        'anomaly_maps': anomaly_maps,
        'cumulative_anomaly': cumulative_anomaly,
        'transitions': transitions,
        'threshold': float(threshold),
        'max_dk': float(max(anomaly_scores)) if anomaly_scores else 0,
    }


# ==========================================================================
# MODULE 2: Radiomics
# ==========================================================================

def kspace_radiomics(images):
    """Extract radiomics features from K-space."""
    all_features = []

    for idx, img in enumerate(images):
        kspace = compute_kspace(img)
        magnitude = np.abs(kspace)
        log_mag = np.log(magnitude + 1)

        features = {'slice': idx + 1}

        features['mean'] = float(np.mean(log_mag))
        features['std'] = float(np.std(log_mag))
        features['skewness'] = float(stats.skew(log_mag.flatten()))
        features['kurtosis'] = float(stats.kurtosis(log_mag.flatten()))

        rows, cols = magnitude.shape
        cy, cx = rows // 2, cols // 2

        # Energy distribution
        total_energy = np.sum(magnitude ** 2)
        center_radius = int(min(cy, cx) * 0.1)
        Y, X = np.ogrid[:rows, :cols]
        center_mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= center_radius ** 2
        center_energy = np.sum(magnitude[center_mask] ** 2)
        features['center_energy_ratio'] = float(center_energy / (total_energy + 1e-10))

        # Entropy
        probs = magnitude.flatten() / (np.sum(magnitude) + 1e-10)
        probs = probs[probs > 0]
        features['entropy'] = float(-np.sum(probs * np.log2(probs)))

        # Radial energy profile
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
        max_r = min(cy, cx)
        radial_energy = []
        for r in range(max_r):
            ring = dist == r
            if np.any(ring):
                radial_energy.append(float(np.mean(magnitude[ring] ** 2)))
        features['radial_energy'] = radial_energy

        # Asymmetry
        left = magnitude[:, :cx]
        right = np.fliplr(magnitude[:, cx:cx + left.shape[1]])
        min_c = min(left.shape[1], right.shape[1])
        features['lr_asymmetry'] = float(np.mean(np.abs(left[:, :min_c] - right[:, :min_c])) /
                                         (np.mean(magnitude) + 1e-10))

        top = magnitude[:cy, :]
        bottom = np.flipud(magnitude[cy:cy + top.shape[0], :])
        min_r = min(top.shape[0], bottom.shape[0])
        features['tb_asymmetry'] = float(np.mean(np.abs(top[:min_r, :] - bottom[:min_r, :])) /
                                         (np.mean(magnitude) + 1e-10))

        # Angular energy
        angles = np.arctan2(Y - cy, X - cx)
        n_sectors = 8
        angular_energy = []
        for s in range(n_sectors):
            a_start = -np.pi + s * (2 * np.pi / n_sectors)
            a_end = a_start + (2 * np.pi / n_sectors)
            sector = (angles >= a_start) & (angles < a_end)
            angular_energy.append(float(np.mean(magnitude[sector] ** 2)))
        features['angular_uniformity'] = float(np.log(np.std(angular_energy) + 1e-10))

        all_features.append(features)

    return all_features


# ==========================================================================
# MODULE 3: Phase Coherence
# ==========================================================================

def phase_coherence_analysis(images):
    """Analyze phase structure in K-space."""
    all_results = []

    for idx, img in enumerate(images):
        kspace = compute_kspace(img)
        phase = np.angle(kspace)
        magnitude = np.abs(kspace)
        result = {'slice': idx + 1}

        # Local Phase Coherence
        phase_grad_y = np.angle(np.exp(1j * np.diff(phase, axis=0)))
        phase_grad_x = np.angle(np.exp(1j * np.diff(phase, axis=1)))
        result['phase_gradient_mean'] = float((np.mean(np.abs(phase_grad_y)) +
                                                np.mean(np.abs(phase_grad_x))) / 2)

        # Hermitian Symmetry
        phase_flipped = np.flip(np.flip(phase, axis=0), axis=1)
        symmetry_error = np.angle(np.exp(1j * (phase + phase_flipped)))
        result['hermitian_error'] = float(np.mean(np.abs(symmetry_error)))

        # Radial Phase Coherence
        rows, cols = phase.shape
        cy, cx = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
        max_r = min(cy, cx)

        radial_coherence = []
        for r in range(1, max_r):
            ring = dist == r
            if np.any(ring):
                coherence = np.abs(np.mean(np.exp(1j * phase[ring])))
                radial_coherence.append(float(coherence))
        result['mean_coherence'] = float(np.mean(radial_coherence)) if radial_coherence else 0
        result['radial_coherence'] = radial_coherence

        # Phase Entropy
        phase_bins = np.histogram(phase.flatten(), bins=256, range=(-np.pi, np.pi))[0]
        phase_probs = phase_bins / (np.sum(phase_bins) + 1e-10)
        phase_probs = phase_probs[phase_probs > 0]
        result['phase_entropy'] = float(-np.sum(phase_probs * np.log2(phase_probs)))

        # Quadrant consistency
        q1 = phase[:cy, cx:]
        q2 = phase[:cy, :cx]
        q3 = phase[cy:, :cx]
        q4 = phase[cy:, cx:]
        min_qr = min(q1.shape[0], q3.shape[0])
        min_qc = min(q1.shape[1], q2.shape[1])
        q_means = [float(np.mean(q[:min_qr, :min_qc])) for q in [q1, q2, q3, q4]]
        result['quadrant_consistency'] = float(np.std(q_means))

        # Store maps for visualization
        result['phase_map'] = phase
        result['symmetry_error_map'] = symmetry_error

        all_results.append(result)

    return all_results


# ==========================================================================
# VISUALIZATION
# ==========================================================================

def generate_plots(images, diff_results, radiomics, phase_results, output_dir):
    """Generate all analysis plots and return paths."""
    plots = {}

    # 1. Anomaly Summary
    plots['summary'] = _plot_summary(images, diff_results, radiomics, output_dir)

    # 2. Differential Map
    plots['differential'] = _plot_differential(diff_results, len(images), output_dir)

    # 3. Radiomics Report
    plots['radiomics'] = _plot_radiomics(radiomics, output_dir)

    # 4. Phase Coherence
    plots['phase'] = _plot_phase(phase_results, output_dir)

    # 5. Slice gallery (sample images)
    plots['gallery'] = _plot_gallery(images, output_dir)

    return plots


def _plot_summary(images, diff_results, radiomics, output_dir):
    """Generate summary overview plot."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('K-Space Analysis Summary', fontsize=16, fontweight='bold')

    # Original image (middle slice)
    mid = len(images) // 2
    axes[0, 0].imshow(images[mid], cmap='gray')
    axes[0, 0].set_title(f'Original (Slice {mid + 1})')
    axes[0, 0].axis('off')

    # K-Space
    kspace = compute_kspace(images[mid])
    axes[0, 1].imshow(np.log(np.abs(kspace) + 1), cmap='gray')
    axes[0, 1].set_title('K-Space Magnitude')
    axes[0, 1].axis('off')

    # Cumulative anomaly
    if diff_results['cumulative_anomaly'] is not None:
        im = axes[0, 2].imshow(diff_results['cumulative_anomaly'], cmap='hot')
        axes[0, 2].set_title('Cumulative Anomaly Map')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # dK scores
    if diff_results['anomaly_scores']:
        scores = diff_results['anomaly_scores']
        axes[1, 0].bar(range(1, len(scores) + 1), scores, color='steelblue')
        axes[1, 0].axhline(y=diff_results['threshold'], color='red', linestyle='--', label='Threshold')
        axes[1, 0].set_title('dK Scores per Transition')
        axes[1, 0].set_xlabel('Transition')
        axes[1, 0].legend()

    # Entropy trend
    entropies = [f['entropy'] for f in radiomics]
    axes[1, 1].plot(range(1, len(entropies) + 1), entropies, 'o-', color='green')
    axes[1, 1].set_title('Entropy per Slice')
    axes[1, 1].set_xlabel('Slice')

    # LR Asymmetry
    asym = [f['lr_asymmetry'] for f in radiomics]
    axes[1, 2].plot(range(1, len(asym) + 1), asym, 's-', color='purple')
    axes[1, 2].set_title('LR Asymmetry per Slice')
    axes[1, 2].set_xlabel('Slice')

    plt.tight_layout()
    path = os.path.join(output_dir, 'summary.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return 'summary.png'


def _plot_differential(diff_results, n_slices, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Differential Analysis (dK)', fontsize=14, fontweight='bold')

    if diff_results['anomaly_scores']:
        scores = diff_results['anomaly_scores']
        colors = ['red' if t['is_anomaly'] else 'steelblue' for t in diff_results['transitions']]
        axes[0].bar(range(1, len(scores) + 1), scores, color=colors)
        axes[0].axhline(y=diff_results['threshold'], color='red', linestyle='--')
        axes[0].set_title('dK Scores')
        axes[0].set_xlabel('Transition')

    if diff_results['cumulative_anomaly'] is not None:
        im = axes[1].imshow(diff_results['cumulative_anomaly'], cmap='hot')
        axes[1].set_title('Cumulative Anomaly')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

    if diff_results['delta_magnitudes']:
        mid = len(diff_results['delta_magnitudes']) // 2
        worst_idx = np.argmax(diff_results['anomaly_scores'])
        im = axes[2].imshow(np.log(diff_results['delta_magnitudes'][worst_idx] + 1), cmap='inferno')
        axes[2].set_title(f'Worst dK (Slice {worst_idx + 1}->{worst_idx + 2})')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    path = os.path.join(output_dir, 'differential.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return 'differential.png'


def _plot_radiomics(radiomics, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('K-Space Radiomics', fontsize=14, fontweight='bold')
    slices = [f['slice'] for f in radiomics]

    metrics = [
        ('entropy', 'Entropy', 'green'),
        ('lr_asymmetry', 'LR Asymmetry', 'purple'),
        ('tb_asymmetry', 'TB Asymmetry', 'orange'),
        ('center_energy_ratio', 'Center Energy Ratio', 'red'),
        ('skewness', 'Skewness', 'blue'),
        ('angular_uniformity', 'Angular Uniformity', 'teal'),
    ]

    for ax, (key, title, color) in zip(axes.flatten(), metrics):
        values = [f[key] for f in radiomics]
        ax.plot(slices, values, 'o-', color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Slice')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'radiomics.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return 'radiomics.png'


def _plot_phase(phase_results, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Phase Coherence Analysis', fontsize=14, fontweight='bold')

    mid = len(phase_results) // 2
    slices = [r['slice'] for r in phase_results]

    # Phase map
    im = axes[0, 0].imshow(phase_results[mid]['phase_map'], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 0].set_title(f'Phase Map (Slice {mid + 1})')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    # Symmetry error
    err = phase_results[mid]['symmetry_error_map']
    im = axes[0, 1].imshow(np.abs(err), cmap='hot', vmin=0, vmax=np.pi)
    axes[0, 1].set_title(f'Hermitian Error (Slice {mid + 1})')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # Radial coherence profile
    if phase_results[mid]['radial_coherence']:
        axes[0, 2].plot(phase_results[mid]['radial_coherence'], color='cyan', linewidth=1.5)
        axes[0, 2].set_title('Radial Phase Coherence')
        axes[0, 2].set_xlabel('Radius')
        axes[0, 2].set_ylabel('Coherence')

    # Trends
    trends = [
        ('phase_gradient_mean', 'Phase Gradient', 'orange'),
        ('hermitian_error', 'Hermitian Error', 'red'),
        ('mean_coherence', 'Mean Coherence', 'cyan'),
    ]
    for ax, (key, title, color) in zip(axes[1], trends):
        values = [r[key] for r in phase_results]
        ax.plot(slices, values, 'o-', color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Slice')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'phase.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return 'phase.png'


def _plot_gallery(images, output_dir):
    """Show sample slices from the series."""
    n = min(8, len(images))
    indices = np.linspace(0, len(images) - 1, n, dtype=int)

    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
    if n == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        ax.imshow(images[idx], cmap='gray')
        ax.set_title(f'Slice {idx + 1}', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, 'gallery.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return 'gallery.png'


# ==========================================================================
# FULL ANALYSIS PIPELINE
# ==========================================================================

def run_full_analysis(dicom_path, output_dir):
    """
    Run the complete K-Space analysis pipeline.
    Returns structured results dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load
    images, metadata = load_dicom_series(dicom_path)
    if not images:
        return {'error': 'No DICOM images found'}

    # Analyze
    diff_results = differential_analysis(images)
    radiomics = kspace_radiomics(images)
    phase_results = phase_coherence_analysis(images)
    
    # ----------------------------------------------------
    # NEW: AI VIRTUAL BIOPSY & IMAGE-FREE DETECTION
    # ----------------------------------------------------
    ai_fingerprints = []
    for img in images:
        kspace = compute_kspace(img)
        fp = generate_kspace_fingerprint(np.abs(kspace), np.angle(kspace))
        ai_fingerprints.append(fp)
        
    # For a simulated baseline, we use the average of the first two and last two slices 
    # (assuming they are usually normal tissue)
    # If the series is too short, just use the first slice.
    if len(ai_fingerprints) >= 4:
        baseline_fp = np.mean([ai_fingerprints[0], ai_fingerprints[1], ai_fingerprints[-1], ai_fingerprints[-2]], axis=0)
    elif len(ai_fingerprints) > 0:
        baseline_fp = ai_fingerprints[0]
    else:
        baseline_fp = []
        
    ai_predictions = []
    if len(baseline_fp) > 0:
        ai_predictions = image_free_pathology_detection(ai_fingerprints, baseline_fingerprint=baseline_fp)
    
    # Generate plots
    plots = generate_plots(images, diff_results, radiomics, phase_results, output_dir)

    # Build summary metrics
    summary = {
        'slice_count': len(images),
        'image_size': f"{images[0].shape[0]}x{images[0].shape[1]}",
        'dk_max': diff_results['max_dk'],
        'dk_threshold': diff_results['threshold'],
        'anomalies_found': sum(1 for t in diff_results['transitions'] if t['is_anomaly']),
        'entropy_mean': float(np.mean([f['entropy'] for f in radiomics])),
        'entropy_range': [float(min(f['entropy'] for f in radiomics)),
                          float(max(f['entropy'] for f in radiomics))],
        'lr_asymmetry_mean': float(np.mean([f['lr_asymmetry'] for f in radiomics])),
        'phase_coherence_mean': float(np.mean([r['mean_coherence'] for r in phase_results])),
        'hermitian_error_mean': float(np.mean([r['hermitian_error'] for r in phase_results])),
    }

    # Generate Text Report (Turkish)
    summary['report_text'] = generate_text_report(summary, diff_results['transitions'])

    # Clean serializable results (remove numpy arrays)

    # Clean serializable results (remove numpy arrays)
    clean_radiomics = []
    for f in radiomics:
        clean = {k: v for k, v in f.items() if k != 'radial_energy'}
        clean_radiomics.append(clean)

    clean_phase = []
    for r in phase_results:
        clean = {k: v for k, v in r.items()
                 if k not in ('radial_coherence', 'phase_map', 'symmetry_error_map')}
        clean_phase.append(clean)

    return {
        'metadata': metadata,
        'summary': summary,
        'transitions': diff_results['transitions'],
        'radiomics': clean_radiomics,
        'phase': clean_phase,
        'plots': plots,
        'ai_insights': {
            'fingerprints': ai_fingerprints,
            'predictions': ai_predictions
        }
    }


# ==========================================================================
# NEW FEATURES: COMPARISON & 3D
# ==========================================================================

def compare_studies(id1, id2, images1, images2, output_dir):
    """
    Compare two studies (e.g. T1 vs T1+C).
    Returns metrics and paths to comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Align series (simple resizing to match the larger one)
    target_shape = max(images1[0].shape, images2[0].shape)
    
    def resize_stack(imgs, shape):
        resized = []
        for img in imgs:
            if img.shape != shape:
                scale_r = shape[0] / img.shape[0]
                scale_c = shape[1] / img.shape[1]
                resized.append(zoom(img, (scale_r, scale_c), order=1))
            else:
                resized.append(img)
        return resized

    # Sync slice counts (take min)
    n_slices = min(len(images1), len(images2))
    imgs1 = resize_stack(images1[:n_slices], target_shape)
    imgs2 = resize_stack(images2[:n_slices], target_shape)
    
    # 2. Compute differences
    diff_metrics = []
    
    for i in range(n_slices):
        k1 = compute_kspace(imgs1[i])
        k2 = compute_kspace(imgs2[i])
        
        # Magnitude difference
        mag_diff = np.abs(np.abs(k1) - np.abs(k2))
        
        # Phase difference
        phase_diff = np.angle(np.exp(1j * (np.angle(k1) - np.angle(k2))))
        
        diff_metrics.append({
            'slice': i + 1,
            'mean_mag_diff': float(np.mean(mag_diff)),
            'mean_phase_diff': float(np.mean(np.abs(phase_diff)))
        })

    # 3. Generate Comparison Plot (Middle Slice)
    mid = n_slices // 2
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Comparison: {id1} vs {id2} (Slice {mid+1})', fontsize=16)
    
    # Images
    axes[0,0].imshow(imgs1[mid], cmap='gray')
    axes[0,0].set_title(f'Study {id1}')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(imgs2[mid], cmap='gray')
    axes[0,1].set_title(f'Study {id2}')
    axes[0,1].axis('off')
    
    # Difference Image (Subtraction)
    sub = np.abs(imgs1[mid] - imgs2[mid])
    axes[0,2].imshow(sub, cmap='hot')
    axes[0,2].set_title('Image Subtraction')
    axes[0,2].axis('off')
    
    # K-Space Differences
    k1 = compute_kspace(imgs1[mid])
    k2 = compute_kspace(imgs2[mid])
    dk = np.abs(np.abs(k1) - np.abs(k2))
    
    im = axes[1,0].imshow(np.log(dk + 1), cmap='RdBu_r')
    axes[1,0].set_title('K-Space Mag Diff')
    axes[1,0].axis('off')
    plt.colorbar(im, ax=axes[1,0])
    
    dp = np.angle(np.exp(1j * (np.angle(k1) - np.angle(k2))))
    im = axes[1,1].imshow(dp, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1,1].set_title('Phase Diff')
    axes[1,1].axis('off')
    plt.colorbar(im, ax=axes[1,1])
    
    # Metrics
    m_diffs = [m['mean_mag_diff'] for m in diff_metrics]
    axes[1,2].plot(range(1, n_slices+1), m_diffs, 'r-o')
    axes[1,2].set_title('Mean K-Space Diff per Slice')
    axes[1,2].set_xlabel('Slice')
    
    plt.tight_layout()
    plot_name = f'compare_{id1}_{id2}.png'
    plt.savefig(os.path.join(output_dir, plot_name))
    plt.close()
    
    return {
        'metrics': diff_metrics,
        'plot': plot_name,
        'slice_count': n_slices
    }


def generate_3d_points(images, threshold_std=2.0):
    """
    Generate 3D point cloud data for anomalies.
    Returns list of {x, y, z, intensity, color}.
    """
    points = []
    
    # Analyze each slice for anomalies
    # We use high-pass filtered energy as a proxy for "anomalous edge activity"
    # or simple pixel intensity for now, but K-Space anomaly map is better.
    
    for z, img in enumerate(images):
        # 1. Compute anomaly map for this slice (e.g. high freq content)
        kspace = compute_kspace(img)
        rows, cols = kspace.shape
        cy, cx = rows // 2, cols // 2
        
        # High pass filter
        Y, X = np.ogrid[:rows, :cols]
        radius = min(cy, cx) * 0.2
        mask = ((X - cx)**2 + (Y - cy)**2) > radius**2
        
        # Reconstruct high freq only
        k_hp = kspace * mask
        recon_hp = np.abs(np.fft.ifft2(np.fft.ifftshift(k_hp)))
        
        # Threshold
        mean = np.mean(recon_hp)
        std = np.std(recon_hp)
        thresh = mean + threshold_std * std
        
        # Downsample for performance (optional, but good for web 3D)
        # stepping = 2
        
        for y in range(0, rows, 4):
            for x in range(0, cols, 4):
                val = float(recon_hp[y, x])
                if val > thresh:
                    points.append({
                        'x': x,
                        'y': y,
                        'z': z * 5, # Exaggerate Z spacing for visualization
                        'val': round(val, 2)
                    })
                    
    return points


def generate_text_report(summary, transitions):
    """
    Generate a human-readable report in Turkish based on analysis metrics.
    """
    lines = []
    lines.append("### 📋 K-Space Analiz Raporu")
    lines.append(f"**Tarih:** {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    lines.append(f"**İncelenen Kesit Sayısı:** {summary['slice_count']}")
    
    # 1. Anomali Durumu
    anomalies = [t for t in transitions if t['is_anomaly']]
    if anomalies:
        lines.append(f"\n#### 🚨 Tespit Edilen Anomaliler ({len(anomalies)} Adet)")
        lines.append("K-Space frekans analizinde, normal doku yapısından sapmalar tespit edilmiştir. Bu bölgeler patoloji şüphesi taşıyabilir:")
        for a in anomalies:
            lines.append(f"- **Kesit {a['from_slice']} -> {a['to_slice']}**: dK Skoru {a['dk_score']} (Eşik: {summary['dk_threshold']:.2f})")
            lines.append("  > *Yorum:* Bu geçişte yüksek frekans değişimi var. Doku sınırlarında veya yapısında ani değişim gözleniyor.")
    else:
        lines.append("\n#### ✅ Normal Bulgular")
        lines.append("İncelenen seride belirgin bir K-Space anomalisi (dK > Eşik) tespit edilmemiştir.")

    # 2. Faz Uyumu (Phase Coherence)
    coh = summary['phase_coherence_mean']
    lines.append("\n#### 🌊 Faz Uyumu (Phase Coherence)")
    lines.append(f"**Ortalama Skor:** {coh:.4f}")
    if coh < 0.2:
        lines.append("> *Yorum:* Düşük faz uyumu. Görüntüde karmaşık doku yapıları veya gürültü baskın olabilir.")
    else:
        lines.append("> *Yorum:* Yüksek faz uyumu. Doku yapısı düzenli ve homojen görünüyor.")

    # 3. Radyomik Özet
    ent = summary['entropy_mean']
    lines.append("\n#### 🔬 Radyomik Doku Analizi")
    lines.append(f"**Entropi (Karmaşıklık):** {ent:.2f}")
    lines.append("> *Bilgi:* Yüksek entropi, daha karmaşık ve heterojen doku yapısını (örn. tümör dokusu) işaret edebilir.")
    
    asym = summary['lr_asymmetry_mean']
    lines.append(f"**Asimetri İndeksi:** {asym:.4f}")
    if asym > 0.1:
        lines.append("> *Dikkat:* Beyin hemisferleri arasında belirgin asimetri var. Kitle etkisi olabilir.")
    
    lines.append("\n---\n*Bu rapor yapay zeka destekli K-Space analizi ile oluşturulmuştur. Kesin tanı için radyolog yorumu esastır.*")
    
    return "\n".join(lines)
