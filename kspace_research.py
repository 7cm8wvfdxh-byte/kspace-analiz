"""
K-Space Research Analysis Tool
==============================
Radyoloji araştırması için K-space doğrudan analiz aracı.

Modüller:
  1. K-Space Differential Analysis (dK) - Kesitler arasi frekans anomali tespiti
  2. K-Space Radiomics - K-space uzerinde istatistiksel ozellik cikarimi
  3. Supporting Metrics - Enerji profili, entropi, asimetri

Kullanım:
  python kspace_research.py DICOMDIR
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid plt.show() blocking
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pydicom
import os
import sys
import csv
from scipy import stats as scipy_stats

# ============================================================================
# DICOM LOADING
# ============================================================================

def load_dicom_series(path):
    """Loads all DICOM images from a DICOMDIR, directory, or single file, sorted by InstanceNumber."""
    images = []
    
    if os.path.basename(path) == "DICOMDIR":
        print("[DIR] DICOMDIR detected. Scanning...")
        ds = pydicom.dcmread(path)
        base_dir = os.path.dirname(path) if os.path.dirname(path) else '.'
        
        image_records = []
        for record in ds.DirectoryRecordSequence:
            if record.DirectoryRecordType == "IMAGE":
                parts = record.ReferencedFileID
                if isinstance(parts, str):
                    parts = [parts]
                image_path = os.path.join(base_dir, *parts)
                try:
                    dcm = pydicom.dcmread(image_path)
                    inst = int(dcm.get('InstanceNumber', 0))
                    image_records.append((inst, dcm.pixel_array.astype(float)))
                except Exception as e:
                    print(f"  [!] Skipped: {image_path} ({e})")
        
        image_records.sort(key=lambda x: x[0])
        images = [img for _, img in image_records]
        print(f"  [OK] Loaded {len(images)} slices")
    elif os.path.isdir(path):
        print(f"[DIR] Directory detected. Scanning: {path}")
        image_records = []
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath):
                try:
                    dcm = pydicom.dcmread(fpath)
                    inst = int(dcm.get('InstanceNumber', 0))
                    image_records.append((inst, dcm.pixel_array.astype(float)))
                except Exception as e:
                    print(f"  [!] Skipped: {fpath} ({e})")
        image_records.sort(key=lambda x: x[0])
        images = [img for _, img in image_records]
        print(f"  [OK] Loaded {len(images)} slices from directory")
    else:
        # Single file
        dcm = pydicom.dcmread(path)
        images = [dcm.pixel_array.astype(float)]
        print(f"  [OK] Loaded single file")
    
    return images

# ============================================================================
# MODULE 1: K-SPACE DIFFERENTIAL ANALYSIS (ΔK)
# ============================================================================

def compute_kspace(image):
    """Compute centered K-space from image."""
    return np.fft.fftshift(np.fft.fft2(image))

def differential_analysis(images):
    """
    Compute ΔK between consecutive slices.
    Returns:
      - delta_magnitudes: list of |ΔK| maps
      - anomaly_scores: per-slice anomaly score (how much ΔK deviates from expected)
      - anomaly_map: 2D map showing accumulated anomalies across all slices
    """
    print("\n[MODULE 1] K-Space Differential Analysis")
    print("=" * 50)
    
    kspaces = [compute_kspace(img) for img in images]
    
    delta_magnitudes = []
    delta_phases = []
    anomaly_scores = []
    
    # Compute ΔK for each consecutive pair
    for i in range(len(kspaces) - 1):
        dk = kspaces[i+1] - kspaces[i]
        dk_mag = np.abs(dk)
        dk_phase = np.angle(dk)
        
        delta_magnitudes.append(dk_mag)
        delta_phases.append(dk_phase)
    
    # Compute global statistics for anomaly detection
    # Stack all ΔK magnitudes and compute mean/std per pixel
    dk_stack = np.stack(delta_magnitudes, axis=0)  # (N-1, H, W)
    global_mean = np.mean(dk_stack, axis=0)
    global_std = np.std(dk_stack, axis=0) + 1e-10  # avoid division by zero
    
    # Z-score for each slice pair's ΔK
    anomaly_maps = []
    for i, dk_mag in enumerate(delta_magnitudes):
        z_score = (dk_mag - global_mean) / global_std
        anomaly_score = np.mean(np.abs(z_score))  # overall anomaly score
        anomaly_scores.append(anomaly_score)
        anomaly_maps.append(z_score)
        
        status = "[!] ANOMALY" if anomaly_score > 2.0 else "[OK] Normal"
        print(f"  Slice {i+1}->{i+2}: dK Score = {anomaly_score:.3f} {status}")
    
    # Cumulative anomaly heatmap
    cumulative_anomaly = np.max(np.abs(np.stack(anomaly_maps, axis=0)), axis=0)
    
    return {
        'delta_magnitudes': delta_magnitudes,
        'delta_phases': delta_phases,
        'anomaly_scores': anomaly_scores,
        'anomaly_maps': anomaly_maps,
        'cumulative_anomaly': cumulative_anomaly
    }

# ============================================================================
# MODULE 2: K-SPACE RADIOMICS
# ============================================================================

def kspace_radiomics(images):
    """
    Extract radiomic features directly from K-space data.
    Returns a list of feature dictionaries, one per slice.
    """
    print("\n[MODULE 2] K-Space Radiomics")
    print("=" * 50)
    
    all_features = []
    
    for idx, img in enumerate(images):
        kspace = compute_kspace(img)
        kspace_mag = np.abs(kspace)
        kspace_log = np.log(kspace_mag + 1)
        
        features = {}
        features['slice'] = idx + 1
        
        # ---- First-Order Statistics ----
        features['mean'] = np.mean(kspace_mag)
        features['std'] = np.std(kspace_mag)
        features['skewness'] = float(scipy_stats.skew(kspace_mag.flatten()))
        features['kurtosis'] = float(scipy_stats.kurtosis(kspace_mag.flatten()))
        features['energy'] = np.sum(kspace_mag ** 2)
        
        # Shannon Entropy
        hist, _ = np.histogram(kspace_log.flatten(), bins=256, density=True)
        hist = hist[hist > 0]
        features['entropy'] = -np.sum(hist * np.log2(hist))
        
        # ---- Radial Energy Profile ----
        rows, cols = kspace_mag.shape
        cy, cx = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
        max_radius = min(cy, cx)
        
        radial_energy = []
        for r in range(max_radius):
            ring = kspace_mag[dist == r]
            if len(ring) > 0:
                radial_energy.append(np.mean(ring ** 2))
            else:
                radial_energy.append(0)
        
        features['radial_energy'] = radial_energy
        
        # Radial energy concentration: ratio of center energy to total
        center_radius = max_radius // 5  # inner 20%
        center_energy = sum(radial_energy[:center_radius])
        total_energy = sum(radial_energy)
        features['center_ratio'] = center_energy / (total_energy + 1e-10)
        
        # ---- Quadrant Analysis (Asymmetry) ----
        q1 = kspace_mag[:cy, :cx]   # Top-left
        q2 = kspace_mag[:cy, cx:]   # Top-right
        q3 = kspace_mag[cy:, :cx]   # Bottom-left
        q4 = kspace_mag[cy:, cx:]   # Bottom-right
        
        # Ensure same size for comparison
        min_r = min(q1.shape[0], q2.shape[0], q3.shape[0], q4.shape[0])
        min_c = min(q1.shape[1], q2.shape[1], q3.shape[1], q4.shape[1])
        q1, q2, q3, q4 = q1[:min_r,:min_c], q2[:min_r,:min_c], q3[:min_r,:min_c], q4[:min_r,:min_c]
        
        # Left-Right asymmetry
        left = (q1 + q3) / 2
        right = (q2 + q4) / 2
        features['lr_asymmetry'] = np.mean(np.abs(left - right)) / (np.mean(left + right) / 2 + 1e-10)
        
        # Top-Bottom asymmetry
        top = (q1 + q2) / 2
        bottom = (q3 + q4) / 2
        features['tb_asymmetry'] = np.mean(np.abs(top - bottom)) / (np.mean(top + bottom) / 2 + 1e-10)
        
        # ---- Angular Profile ----
        Y_grid, X_grid = np.mgrid[:rows, :cols]
        angles = np.arctan2(Y_grid - cy, X_grid - cx)  # -pi to pi
        n_bins = 36  # 10-degree bins
        angular_energy = []
        angle_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        for b in range(n_bins):
            mask = (angles >= angle_bins[b]) & (angles < angle_bins[b+1])
            if np.any(mask):
                angular_energy.append(np.mean(kspace_mag[mask] ** 2))
            else:
                angular_energy.append(0)
        
        features['angular_energy'] = angular_energy
        
        # Angular uniformity (high = isotropic, low = directional texture)
        ang_arr = np.array(angular_energy)
        features['angular_uniformity'] = 1.0 - (np.std(ang_arr) / (np.mean(ang_arr) + 1e-10))
        
        all_features.append(features)
        print(f"  Slice {idx+1}: Entropy={features['entropy']:.2f}, "
              f"CenterRatio={features['center_ratio']:.3f}, "
              f"LR_Asym={features['lr_asymmetry']:.4f}, "
              f"AngUnif={features['angular_uniformity']:.3f}")
    
    return all_features

# ============================================================================
# MODULE 3: K-SPACE PHASE COHERENCE ANALYSIS
# ============================================================================

def phase_coherence_analysis(images):
    """
    Analyzes the phase structure of K-space data.
    Phase carries information about spatial position and tissue symmetry.
    Pathology disrupts local phase relationships.
    
    Returns per-slice phase metrics and coherence maps.
    """
    print("\n[MODULE 3] K-Space Phase Coherence Analysis")
    print("=" * 50)
    
    all_phase_results = []
    
    for idx, img in enumerate(images):
        kspace = compute_kspace(img)
        phase = np.angle(kspace)  # Phase map: -pi to pi
        
        result = {'slice': idx + 1}
        
        # ---- 1. Local Phase Coherence ----
        # How consistent is the phase between neighboring pixels?
        # High coherence = smooth structure, Low coherence = disruption
        # Compute phase gradient (rate of phase change)
        phase_grad_y = np.diff(phase, axis=0)  # Vertical gradient
        phase_grad_x = np.diff(phase, axis=1)  # Horizontal gradient
        
        # Wrap phase differences to [-pi, pi]
        phase_grad_y = np.angle(np.exp(1j * phase_grad_y))
        phase_grad_x = np.angle(np.exp(1j * phase_grad_x))
        
        # Local coherence = 1 - normalized gradient magnitude
        grad_mag_y = np.abs(phase_grad_y)
        grad_mag_x = np.abs(phase_grad_x)
        
        # Average gradient magnitude (lower = more coherent)
        result['phase_gradient_mean'] = (np.mean(grad_mag_y) + np.mean(grad_mag_x)) / 2
        result['phase_gradient_std'] = (np.std(grad_mag_y) + np.std(grad_mag_x)) / 2
        
        # ---- 2. Phase Symmetry (Hermitian Symmetry) ----
        # For a real-valued image, K-space should have Hermitian symmetry:
        # K(kx, ky) = conj(K(-kx, -ky))
        # This means phase(kx,ky) = -phase(-kx,-ky)
        # Deviation from this indicates asymmetry in the image
        rows, cols = phase.shape
        phase_flipped = np.flip(np.flip(phase, axis=0), axis=1)
        
        # Hermitian symmetry error
        symmetry_error = np.angle(np.exp(1j * (phase + phase_flipped)))  # Should be ~0
        result['hermitian_error_mean'] = np.mean(np.abs(symmetry_error))
        result['hermitian_error_std'] = np.std(np.abs(symmetry_error))
        
        # ---- 3. Radial Phase Coherence ----
        # Phase coherence as a function of distance from center
        cy, cx = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
        max_radius = min(cy, cx)
        
        radial_coherence = []
        for r in range(1, max_radius):
            ring_mask = dist == r
            if np.any(ring_mask):
                ring_phase = phase[ring_mask]
                # Phase coherence = magnitude of mean complex exponential
                # (1.0 = perfectly coherent, 0.0 = completely random)
                coherence = np.abs(np.mean(np.exp(1j * ring_phase)))
                radial_coherence.append(coherence)
            else:
                radial_coherence.append(0)
        
        result['radial_coherence'] = radial_coherence
        result['mean_coherence'] = np.mean(radial_coherence)
        
        # ---- 4. Phase Entropy ----
        # Entropy of the phase distribution (more uniform = more random)
        phase_hist, _ = np.histogram(phase.flatten(), bins=64, range=(-np.pi, np.pi), density=True)
        phase_hist = phase_hist[phase_hist > 0]
        result['phase_entropy'] = -np.sum(phase_hist * np.log2(phase_hist))
        
        # ---- 5. Quadrant Phase Consistency ----
        # Compare phase statistics across quadrants
        q1_phase = phase[:cy, :cx]
        q2_phase = phase[:cy, cx:]
        q3_phase = phase[cy:, :cx]
        q4_phase = phase[cy:, cx:]
        
        q_means = [np.mean(np.abs(q)) for q in [q1_phase, q2_phase, q3_phase, q4_phase]]
        result['phase_quadrant_variance'] = np.var(q_means)
        
        # Store phase map and symmetry error for visualization
        result['phase_map'] = phase
        result['symmetry_error_map'] = symmetry_error
        
        all_phase_results.append(result)
        print(f"  Slice {idx+1}: PhaseGrad={result['phase_gradient_mean']:.4f}, "
              f"Hermitian={result['hermitian_error_mean']:.4f}, "
              f"Coherence={result['mean_coherence']:.4f}, "
              f"PhaseEntropy={result['phase_entropy']:.2f}")
    
    return all_phase_results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_differential_analysis(diff_results, n_slices, output_dir):
    """Generates visualization for ΔK analysis."""
    
    scores = diff_results['anomaly_scores']
    cumulative = diff_results['cumulative_anomaly']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("K-Space Differential Analysis (dK)", fontsize=16, fontweight='bold')
    
    # 1. Anomaly Score per Slice Transition
    ax = axes[0, 0]
    colors = ['red' if s > 2.0 else 'steelblue' for s in scores]
    bars = ax.bar(range(1, len(scores)+1), scores, color=colors)
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Anomaly Threshold (2std)')
    ax.set_xlabel('Slice Transition (n -> n+1)')
    ax.set_ylabel('dK Anomaly Score (Z-Score)')
    ax.set_title('Anomaly Score per Slice Transition')
    ax.legend()
    
    # 2. Cumulative Anomaly Map
    ax = axes[0, 1]
    im = ax.imshow(np.log(cumulative + 1), cmap='hot')
    ax.set_title('Cumulative K-Space Anomaly Map\n(Log Scale)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 3. Show worst ΔK (highest anomaly)
    worst_idx = np.argmax(scores)
    ax = axes[1, 0]
    dk_show = np.log(diff_results['delta_magnitudes'][worst_idx] + 1)
    im = ax.imshow(dk_show, cmap='inferno')
    ax.set_title(f'Worst dK: Slice {worst_idx+1}->{worst_idx+2}\n(Score={scores[worst_idx]:.3f})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 4. ΔK Phase of worst transition
    ax = axes[1, 1]
    phase_show = diff_results['delta_phases'][worst_idx]
    im = ax.imshow(phase_show, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title(f'dK Phase Map: Slice {worst_idx+1}->{worst_idx+2}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'kspace_differential_map.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [SAVED] {path}")

def plot_radiomics(features_list, output_dir):
    """Generates visualization for K-Space Radiomics."""
    
    slices = [f['slice'] for f in features_list]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("K-Space Radiomics Analysis", fontsize=16, fontweight='bold')
    
    # 1. Entropy per Slice
    ax = axes[0, 0]
    entropies = [f['entropy'] for f in features_list]
    ax.plot(slices, entropies, 'o-', color='darkblue', linewidth=2)
    ax.fill_between(slices, entropies, alpha=0.2)
    ax.set_xlabel('Slice')
    ax.set_ylabel('Shannon Entropy')
    ax.set_title('K-Space Entropy per Slice\n(Higher = More Heterogeneous)')
    
    # 2. Center Energy Ratio
    ax = axes[0, 1]
    crs = [f['center_ratio'] for f in features_list]
    ax.plot(slices, crs, 's-', color='darkgreen', linewidth=2)
    ax.fill_between(slices, crs, alpha=0.2, color='green')
    ax.set_xlabel('Slice')
    ax.set_ylabel('Center Energy Ratio')
    ax.set_title('Energy Concentration (Center vs Periphery)\n(Higher = More Contrast-Dominant)')
    
    # 3. LR Asymmetry
    ax = axes[0, 2]
    lr = [f['lr_asymmetry'] for f in features_list]
    tb = [f['tb_asymmetry'] for f in features_list]
    ax.plot(slices, lr, 'D-', color='crimson', linewidth=2, label='Left-Right')
    ax.plot(slices, tb, '^-', color='orange', linewidth=2, label='Top-Bottom')
    ax.set_xlabel('Slice')
    ax.set_ylabel('Asymmetry Index')
    ax.set_title('K-Space Asymmetry\n(Higher = More Asymmetric)')
    ax.legend()
    
    # 4. Radial Energy Profile (middle slice)
    ax = axes[1, 0]
    mid = len(features_list) // 2
    radial = features_list[mid]['radial_energy']
    ax.semilogy(range(len(radial)), radial, color='purple', linewidth=2)
    ax.fill_between(range(len(radial)), radial, alpha=0.15, color='purple')
    ax.set_xlabel('Radius (pixels from center)')
    ax.set_ylabel('Mean Energy (log scale)')
    ax.set_title(f'Radial Energy Profile (Slice {mid+1})\n(K-Space "Fingerprint")')
    
    # 5. Angular Energy Profile (middle slice)
    ax = axes[1, 1]
    angular = features_list[mid]['angular_energy']
    theta = np.linspace(0, 360, len(angular), endpoint=False)
    ax.bar(theta, angular, width=9, color='teal', alpha=0.8)
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Mean Energy')
    ax.set_title(f'Angular Energy Profile (Slice {mid+1})\n(Directional Texture)')
    
    # 6. Multi-feature summary (Skewness + Kurtosis)
    ax = axes[1, 2]
    skew = [f['skewness'] for f in features_list]
    kurt = [f['kurtosis'] for f in features_list]
    ax.scatter(skew, kurt, c=slices, cmap='viridis', s=80, edgecolors='black', zorder=5)
    for i, s in enumerate(slices):
        ax.annotate(str(s), (skew[i], kurt[i]), fontsize=7, ha='center', va='bottom')
    ax.set_xlabel('Skewness')
    ax.set_ylabel('Kurtosis')
    ax.set_title('K-Space Distribution Shape\n(Skewness vs Kurtosis per Slice)')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Slice #')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'kspace_radiomics_report.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [SAVED] {path}")

def plot_anomaly_summary(images, diff_results, features_list, output_dir):
    """Creates a combined anomaly summary visualization."""
    
    n = len(images)
    mid = n // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("K-Space Anomaly Summary - Combined View", fontsize=16, fontweight='bold')
    
    # 1. Original middle slice
    ax = axes[0, 0]
    ax.imshow(images[mid], cmap='gray')
    ax.set_title(f'Original Image (Slice {mid+1})')
    ax.axis('off')
    
    # 2. K-space of middle slice
    ax = axes[0, 1]
    kspace_mid = compute_kspace(images[mid])
    ax.imshow(np.log(np.abs(kspace_mid) + 1), cmap='gray')
    ax.set_title(f'K-Space (Slice {mid+1})')
    ax.axis('off')
    
    # 3. Cumulative Anomaly overlaid on anatomy-like view
    ax = axes[0, 2]
    anomaly = diff_results['cumulative_anomaly']
    ax.imshow(np.log(anomaly + 1), cmap='hot')
    ax.set_title('Cumulative dK Anomaly Map')
    ax.axis('off')
    
    # 4. Entropy trend with anomaly highlights
    ax = axes[1, 0]
    entropies = [f['entropy'] for f in features_list]
    slices = list(range(1, n+1))
    ax.plot(slices, entropies, 'o-', color='navy', linewidth=2)
    # Highlight outliers
    ent_mean = np.mean(entropies)
    ent_std = np.std(entropies)
    for i, e in enumerate(entropies):
        if abs(e - ent_mean) > 1.5 * ent_std:
            ax.scatter(i+1, e, color='red', s=150, zorder=5, marker='*')
    ax.axhline(ent_mean, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Entropy Trend (* = Outliers)')
    ax.set_xlabel('Slice')
    ax.set_ylabel('Entropy')
    
    # 5. dK Score trend
    ax = axes[1, 1]
    scores = diff_results['anomaly_scores']
    trans = list(range(1, len(scores)+1))
    colors = ['red' if s > 2.0 else 'steelblue' for s in scores]
    ax.bar(trans, scores, color=colors)
    ax.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_title('dK Anomaly Scores')
    ax.set_xlabel('Transition')
    ax.set_ylabel('Score')
    ax.legend()
    
    # 6. Asymmetry radar
    ax = axes[1, 2]
    lr_vals = [f['lr_asymmetry'] for f in features_list]
    tb_vals = [f['tb_asymmetry'] for f in features_list]
    ax.scatter(lr_vals, tb_vals, c=slices, cmap='plasma', s=80, edgecolors='black')
    for i in range(n):
        ax.annotate(str(i+1), (lr_vals[i], tb_vals[i]), fontsize=7)
    ax.set_xlabel('Left-Right Asymmetry')
    ax.set_ylabel('Top-Bottom Asymmetry')
    ax.set_title('Asymmetry Space\n(Outliers = Potential Pathology)')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'kspace_anomaly_summary.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [SAVED] {path}")

def save_csv(features_list, output_dir):
    """Save radiomics features to CSV for further analysis."""
    path = os.path.join(output_dir, 'kspace_radiomics_data.csv')
    
    # Exclude array fields from CSV
    exclude_keys = {'radial_energy', 'angular_energy'}
    
    keys = [k for k in features_list[0].keys() if k not in exclude_keys]
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for feat in features_list:
            row = {k: feat[k] for k in keys}
            writer.writerow(row)
    
    print(f"  [SAVED] {path}")

# ============================================================================
# PHASE COHERENCE VISUALIZATION
# ============================================================================

def plot_phase_coherence(phase_results, output_dir):
    """Generates visualization for Phase Coherence Analysis."""
    
    slices = [r['slice'] for r in phase_results]
    n = len(phase_results)
    mid = n // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("K-Space Phase Coherence Analysis", fontsize=16, fontweight='bold')
    
    # 1. Phase Map of middle slice
    ax = axes[0, 0]
    im = ax.imshow(phase_results[mid]['phase_map'], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title(f'Phase Map (Slice {mid+1})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 2. Hermitian Symmetry Error Map
    ax = axes[0, 1]
    err = phase_results[mid]['symmetry_error_map']
    im = ax.imshow(np.abs(err), cmap='hot', vmin=0, vmax=np.pi)
    ax.set_title(f'Hermitian Symmetry Error (Slice {mid+1})\n(Bright = Asymmetric)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 3. Radial Phase Coherence Profile
    ax = axes[0, 2]
    rc = phase_results[mid]['radial_coherence']
    ax.plot(range(len(rc)), rc, color='teal', linewidth=2)
    ax.fill_between(range(len(rc)), rc, alpha=0.2, color='teal')
    ax.set_xlabel('Radius from K-Space Center')
    ax.set_ylabel('Phase Coherence (0-1)')
    ax.set_title(f'Radial Phase Coherence (Slice {mid+1})\n(1=Coherent, 0=Random)')
    ax.set_ylim(0, 1)
    
    # 4. Phase Gradient per Slice
    ax = axes[1, 0]
    grads = [r['phase_gradient_mean'] for r in phase_results]
    ax.plot(slices, grads, 'o-', color='darkred', linewidth=2)
    ax.fill_between(slices, grads, alpha=0.2, color='red')
    ax.set_xlabel('Slice')
    ax.set_ylabel('Mean Phase Gradient')
    ax.set_title('Phase Gradient per Slice\n(Higher = More Phase Disruption)')
    
    # 5. Hermitian Error per Slice
    ax = axes[1, 1]
    herr = [r['hermitian_error_mean'] for r in phase_results]
    ax.plot(slices, herr, 's-', color='purple', linewidth=2)
    ax.fill_between(slices, herr, alpha=0.2, color='purple')
    ax.set_xlabel('Slice')
    ax.set_ylabel('Mean Hermitian Error')
    ax.set_title('Hermitian Symmetry Error per Slice\n(Higher = More Structural Asymmetry)')
    
    # 6. Phase Entropy per Slice
    ax = axes[1, 2]
    pe = [r['phase_entropy'] for r in phase_results]
    ax.plot(slices, pe, 'D-', color='darkorange', linewidth=2)
    ax.fill_between(slices, pe, alpha=0.2, color='orange')
    ax.set_xlabel('Slice')
    ax.set_ylabel('Phase Entropy')
    ax.set_title('Phase Entropy per Slice\n(Higher = More Random Phase)')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'kspace_phase_coherence.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [SAVED] {path}")

def save_phase_csv(phase_results, output_dir):
    """Save phase coherence features to CSV."""
    path = os.path.join(output_dir, 'kspace_phase_data.csv')
    
    exclude_keys = {'radial_coherence', 'phase_map', 'symmetry_error_map'}
    keys = [k for k in phase_results[0].keys() if k not in exclude_keys]
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in phase_results:
            row = {k: r[k] for k in keys}
            writer.writerow(row)
    
    print(f"  [SAVED] {path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python kspace_research.py <DICOMDIR or path> [label]")
        print("  label: optional name for output subfolder (e.g. 'normal', 'pathological')")
        return
    
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else None
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if label:
        output_dir = os.path.join(base_dir, f"results_{label}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = base_dir
    
    print("=" * 60)
    print("  K-SPACE RESEARCH ANALYSIS TOOL")
    print("  Radyoloji Arastirma Araci")
    print("=" * 60)
    
    # Load images
    images = load_dicom_series(path)
    if not images:
        print("No images found!")
        return
    
    # Module 1: Differential Analysis
    diff_results = differential_analysis(images)
    
    # Module 2: Radiomics
    radiomics_features = kspace_radiomics(images)
    
    # Module 3: Phase Coherence
    phase_results = phase_coherence_analysis(images)
    
    # Visualizations
    print("\n[VIZ] Generating Visualizations...")
    print("=" * 50)
    plot_differential_analysis(diff_results, len(images), output_dir)
    plot_radiomics(radiomics_features, output_dir)
    plot_phase_coherence(phase_results, output_dir)
    plot_anomaly_summary(images, diff_results, radiomics_features, output_dir)
    
    # Save data
    save_csv(radiomics_features, output_dir)
    save_phase_csv(phase_results, output_dir)
    
    print("\n[DONE] Analysis complete!")
    print(f"   Output directory: {output_dir}")

if __name__ == "__main__":
    main()
