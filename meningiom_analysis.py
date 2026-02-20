"""
Meningiom Focused K-Space Analysis
===================================
Frontal meningiom bolgesi icin:
  1. K-Space Filtering - Low/High pass ayirimi
  2. Kontrastli (T1+C) vs Kontrastsiz (T1) karsilastirma
  3. Meningiom bolgesi K-Space haritasi
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pydicom
import os
import sys
from scipy.ndimage import zoom


def load_series_from_dir(path):
    """Load and sort DICOM series from directory."""
    records = []
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        if os.path.isfile(fpath):
            try:
                dcm = pydicom.dcmread(fpath)
                inst = int(dcm.get('InstanceNumber', 0))
                records.append((inst, dcm.pixel_array.astype(float)))
            except:
                pass
    records.sort(key=lambda x: x[0])
    return [img for _, img in records]


def compute_kspace(image):
    return np.fft.fftshift(np.fft.fft2(image))


def low_pass_filter(kspace, radius_fraction=0.15):
    """Keep only center frequencies (contrast/structure)."""
    rows, cols = kspace.shape
    cy, cx = rows // 2, cols // 2
    radius = int(min(cy, cx) * radius_fraction)
    Y, X = np.ogrid[:rows, :cols]
    mask = ((X - cx)**2 + (Y - cy)**2) <= radius**2
    return kspace * mask, mask


def high_pass_filter(kspace, radius_fraction=0.15):
    """Keep only peripheral frequencies (edges/details)."""
    rows, cols = kspace.shape
    cy, cx = rows // 2, cols // 2
    radius = int(min(cy, cx) * radius_fraction)
    Y, X = np.ogrid[:rows, :cols]
    mask = ((X - cx)**2 + (Y - cy)**2) > radius**2
    return kspace * mask, mask


def band_pass_filter(kspace, inner_frac=0.05, outer_frac=0.30):
    """Keep mid-range frequencies (texture/pathology details)."""
    rows, cols = kspace.shape
    cy, cx = rows // 2, cols // 2
    r_inner = int(min(cy, cx) * inner_frac)
    r_outer = int(min(cy, cx) * outer_frac)
    Y, X = np.ogrid[:rows, :cols]
    dist = (X - cx)**2 + (Y - cy)**2
    mask = (dist > r_inner**2) & (dist <= r_outer**2)
    return kspace * mask, mask


def reconstruct(kspace):
    """Reconstruct image from K-space."""
    return np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))


def plot_meningiom_filtering(t1_images, t1c_images, meningiom_slice, output_dir):
    """
    Fig 1: K-Space filtering analysis of meningiom slice.
    Shows how meningiom appears in different frequency bands.
    """
    # Use 0-indexed slice
    idx = meningiom_slice - 1
    
    # Make sure index is valid for both series
    idx_t1 = min(idx, len(t1_images) - 1)
    idx_t1c = min(idx, len(t1c_images) - 1)
    
    img_t1 = t1_images[idx_t1]
    img_t1c = t1c_images[idx_t1c]
    
    # Resize T1 to match T1+C if different sizes
    if img_t1.shape != img_t1c.shape:
        scale_r = img_t1c.shape[0] / img_t1.shape[0]
        scale_c = img_t1c.shape[1] / img_t1.shape[1]
        img_t1 = zoom(img_t1, (scale_r, scale_c), order=3)
    
    kspace_t1 = compute_kspace(img_t1)
    kspace_t1c = compute_kspace(img_t1c)
    
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(f"Meningiom K-Space Filtering Analysis (Slice {meningiom_slice})", 
                 fontsize=18, fontweight='bold')
    
    gs = GridSpec(3, 6, figure=fig, hspace=0.35, wspace=0.3)
    
    # Row 1: Original images and K-space
    # T1 Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_t1, cmap='gray')
    ax.set_title('Sag T1\n(Pre-Contrast)', fontsize=10)
    ax.axis('off')
    
    # T1 K-space
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.log(np.abs(kspace_t1) + 1), cmap='gray')
    ax.set_title('K-Space T1', fontsize=10)
    ax.axis('off')
    
    # T1+C Original
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(img_t1c, cmap='gray')
    ax.set_title('Sag T1+C\n(Post-Contrast)', fontsize=10)
    ax.axis('off')
    
    # T1+C K-space
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(np.log(np.abs(kspace_t1c) + 1), cmap='gray')
    ax.set_title('K-Space T1+C', fontsize=10)
    ax.axis('off')
    
    # K-space Difference (T1+C - T1)
    ax = fig.add_subplot(gs[0, 4])
    dk = np.abs(kspace_t1c) - np.abs(kspace_t1)
    # Resize if needed
    min_r = min(dk.shape[0], np.abs(kspace_t1).shape[0])
    min_c = min(dk.shape[1], np.abs(kspace_t1).shape[1])
    im = ax.imshow(np.log(np.abs(dk[:min_r, :min_c]) + 1), cmap='RdBu_r')
    ax.set_title('K-Space Difference\n|T1+C| - |T1|', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Phase difference
    ax = fig.add_subplot(gs[0, 5])
    phase_diff = np.angle(kspace_t1c[:min_r, :min_c]) - np.angle(kspace_t1[:min_r, :min_c])
    phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrap
    im = ax.imshow(phase_diff, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Phase Difference\nT1+C - T1', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 2: T1+C Filtering (meningiom series)
    # Low-pass
    ksp_lp, mask_lp = low_pass_filter(kspace_t1c)
    recon_lp = reconstruct(ksp_lp)
    
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(recon_lp, cmap='gray')
    ax.set_title('Low-Pass Recon\n(Contrast/Structure)', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(np.log(np.abs(ksp_lp) + 1), cmap='gray')
    ax.set_title('Low-Pass K-Space', fontsize=10)
    ax.axis('off')
    
    # Band-pass
    ksp_bp, mask_bp = band_pass_filter(kspace_t1c)
    recon_bp = reconstruct(ksp_bp)
    
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(recon_bp, cmap='gray')
    ax.set_title('Band-Pass Recon\n(Texture/Pathology)', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(np.log(np.abs(ksp_bp) + 1), cmap='gray')
    ax.set_title('Band-Pass K-Space', fontsize=10)
    ax.axis('off')
    
    # High-pass
    ksp_hp, mask_hp = high_pass_filter(kspace_t1c)
    recon_hp = reconstruct(ksp_hp)
    
    ax = fig.add_subplot(gs[1, 4])
    ax.imshow(recon_hp, cmap='gray')
    ax.set_title('High-Pass Recon\n(Edges/Details)', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 5])
    ax.imshow(np.log(np.abs(ksp_hp) + 1), cmap='gray')
    ax.set_title('High-Pass K-Space', fontsize=10)
    ax.axis('off')
    
    # Row 3: T1 Filtering (pre-contrast) for comparison
    ksp_lp_t1, _ = low_pass_filter(kspace_t1)
    recon_lp_t1 = reconstruct(ksp_lp_t1)
    
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(recon_lp_t1, cmap='gray')
    ax.set_title('T1 Low-Pass\n(Pre-Contrast)', fontsize=10)
    ax.axis('off')
    
    # Difference in low-pass (shows contrast enhancement)
    ax = fig.add_subplot(gs[2, 1])
    lp_diff = recon_lp - recon_lp_t1
    im = ax.imshow(lp_diff, cmap='hot')
    ax.set_title('Low-Pass Diff\nContrast Enhancement!', fontsize=10, color='red')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ksp_bp_t1, _ = band_pass_filter(kspace_t1)
    recon_bp_t1 = reconstruct(ksp_bp_t1)
    
    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(recon_bp_t1, cmap='gray')
    ax.set_title('T1 Band-Pass\n(Pre-Contrast)', fontsize=10)
    ax.axis('off')
    
    # Difference in band-pass (shows texture change)
    ax = fig.add_subplot(gs[2, 3])
    bp_diff = recon_bp - recon_bp_t1
    im = ax.imshow(bp_diff, cmap='hot')
    ax.set_title('Band-Pass Diff\nTexture Change!', fontsize=10, color='red')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ksp_hp_t1, _ = high_pass_filter(kspace_t1)
    recon_hp_t1 = reconstruct(ksp_hp_t1)
    
    ax = fig.add_subplot(gs[2, 4])
    ax.imshow(recon_hp_t1, cmap='gray')
    ax.set_title('T1 High-Pass\n(Pre-Contrast)', fontsize=10)
    ax.axis('off')
    
    # Difference in high-pass (shows edge changes)
    ax = fig.add_subplot(gs[2, 5])
    hp_diff = recon_hp - recon_hp_t1
    im = ax.imshow(hp_diff, cmap='hot')
    ax.set_title('High-Pass Diff\nEdge Change!', fontsize=10, color='red')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.savefig(os.path.join(output_dir, 'meningiom_kspace_filtering.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[SAVED] meningiom_kspace_filtering.png")


def plot_contrast_comparison(t1_images, t1c_images, output_dir):
    """
    Fig 2: Multi-slice contrast comparison across the meningiom region.
    """
    fig, axes = plt.subplots(4, 5, figsize=(22, 18))
    fig.suptitle("Meningiom Region: T1 vs T1+C K-Space Slice Comparison (Slice 9-13)", 
                 fontsize=16, fontweight='bold')
    
    meningiom_slices = [9, 10, 11, 12, 13]
    
    for col, s in enumerate(meningiom_slices):
        idx_t1 = min(s - 1, len(t1_images) - 1)
        idx_t1c = min(s - 1, len(t1c_images) - 1)
        
        img_t1 = t1_images[idx_t1]
        img_t1c = t1c_images[idx_t1c]
        
        # Resize T1 to match T1+C
        if img_t1.shape != img_t1c.shape:
            scale_r = img_t1c.shape[0] / img_t1.shape[0]
            scale_c = img_t1c.shape[1] / img_t1.shape[1]
            img_t1 = zoom(img_t1, (scale_r, scale_c), order=3)
        
        ksp_t1 = compute_kspace(img_t1)
        ksp_t1c = compute_kspace(img_t1c)
        
        # Row 1: T1+C images
        axes[0, col].imshow(img_t1c, cmap='gray')
        axes[0, col].set_title(f'Slice {s} T1+C', fontsize=9)
        axes[0, col].axis('off')
        
        # Row 2: K-space magnitude difference
        mag_diff = np.abs(ksp_t1c) - np.abs(ksp_t1)
        axes[1, col].imshow(np.log(np.abs(mag_diff) + 1), cmap='RdBu_r')
        axes[1, col].set_title(f'Slice {s} |dK|', fontsize=9)
        axes[1, col].axis('off')
        
        # Row 3: Radial energy comparison
        mag_t1 = np.abs(ksp_t1)
        mag_t1c = np.abs(ksp_t1c)
        
        rows_t1c, cols_t1c = mag_t1c.shape
        cy, cx = rows_t1c // 2, cols_t1c // 2
        Y, X = np.ogrid[:rows_t1c, :cols_t1c]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
        max_r = min(cy, cx)
        
        re_t1c = []
        re_t1 = []
        for r in range(max_r):
            ring = dist == r
            if np.any(ring):
                re_t1c.append(np.mean(mag_t1c[ring]**2))
                re_t1.append(np.mean(mag_t1[ring]**2))
        
        axes[2, col].semilogy(re_t1c, color='red', linewidth=1.5, label='T1+C', alpha=0.8)
        axes[2, col].semilogy(re_t1, color='blue', linewidth=1.5, label='T1', alpha=0.8)
        axes[2, col].set_title(f'Slice {s} Radial', fontsize=9)
        axes[2, col].legend(fontsize=7)
        if col == 0:
            axes[2, col].set_ylabel('Energy (log)')
        
        # Row 4: Phase coherence comparison 
        phase_t1 = np.angle(ksp_t1)
        phase_t1c = np.angle(ksp_t1c)
        
        rc_t1c = []
        rc_t1 = []
        for r in range(1, max_r):
            ring = dist == r
            if np.any(ring):
                rc_t1c.append(np.abs(np.mean(np.exp(1j * phase_t1c[ring]))))
                rc_t1.append(np.abs(np.mean(np.exp(1j * phase_t1[ring]))))
        
        axes[3, col].plot(rc_t1c, color='red', linewidth=1.5, label='T1+C', alpha=0.8)
        axes[3, col].plot(rc_t1, color='blue', linewidth=1.5, label='T1', alpha=0.8)
        axes[3, col].set_title(f'Slice {s} Phase Coh', fontsize=9)
        axes[3, col].set_ylim(0, 0.5)
        axes[3, col].legend(fontsize=7)
        if col == 0:
            axes[3, col].set_ylabel('Coherence')
    
    # Add row labels
    for ax, label in zip([axes[0,0], axes[1,0], axes[2,0], axes[3,0]], 
                          ['T1+C Image', 'K-Space |dK|', 'Radial Energy', 'Phase Coherence']):
        ax.annotate(label, xy=(-0.3, 0.5), xycoords='axes fraction', 
                   fontsize=11, fontweight='bold', rotation=90, va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'meningiom_contrast_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[SAVED] meningiom_contrast_comparison.png")


def plot_meningiom_heatmap(t1c_images, output_dir):
    """
    Fig 3: K-Space anomaly heatmap focused on contrast enhancement region.
    Shows where in K-space the meningiom signal is strongest.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Meningiom K-Space Signal Map (Sag T1+C)", fontsize=16, fontweight='bold')
    
    # Meningiom slice (0-indexed: 10 = Slice 11, highest dK)
    men_idx = 10
    img = t1c_images[men_idx]
    kspace = compute_kspace(img)
    mag = np.abs(kspace)
    phase = np.angle(kspace)
    
    # Adjacent normal slice for comparison
    norm_idx = 2  # Far from meningiom
    img_norm = t1c_images[norm_idx]
    kspace_norm = compute_kspace(img_norm)
    mag_norm = np.abs(kspace_norm)
    
    # 1. Original image with meningiom region highlighted
    ax = axes[0, 0]
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Meningiom Slice ({men_idx+1})\nFrontal Region', fontsize=11)
    ax.axis('off')
    
    # 2. Normal slice for comparison
    ax = axes[0, 1]
    ax.imshow(img_norm, cmap='gray')
    ax.set_title(f'Normal Slice ({norm_idx+1})\nReference', fontsize=11)
    ax.axis('off')
    
    # 3. K-space magnitude ratio
    ax = axes[0, 2]
    min_r = min(mag.shape[0], mag_norm.shape[0])
    min_c = min(mag.shape[1], mag_norm.shape[1])
    ratio = mag[:min_r, :min_c] / (mag_norm[:min_r, :min_c] + 1e-10)
    ratio_log = np.log(ratio + 1e-10)
    im = ax.imshow(ratio_log, cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title(f'K-Space Ratio\nMeningiom/Normal (log)', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 4. Local energy map of meningiom slice (windowed)
    ax = axes[1, 0]
    window_size = 16
    rows, cols = mag.shape
    energy_map = np.zeros_like(mag)
    for i in range(0, rows - window_size, window_size // 2):
        for j in range(0, cols - window_size, window_size // 2):
            patch = mag[i:i+window_size, j:j+window_size]
            energy_map[i:i+window_size, j:j+window_size] = np.mean(patch**2)
    im = ax.imshow(np.log(energy_map + 1), cmap='inferno')
    ax.set_title('Local Energy Map\n(Meningiom Slice)', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 5. Phase coherence map (local windows)
    ax = axes[1, 1]
    coh_map = np.zeros_like(phase)
    for i in range(0, rows - window_size, window_size // 2):
        for j in range(0, cols - window_size, window_size // 2):
            patch = phase[i:i+window_size, j:j+window_size]
            local_coh = np.abs(np.mean(np.exp(1j * patch)))
            coh_map[i:i+window_size, j:j+window_size] = local_coh
    im = ax.imshow(coh_map, cmap='viridis', vmin=0, vmax=0.5)
    ax.set_title('Local Phase Coherence Map\n(Meningiom Slice)', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 6. Spectral difference map (what kontrast adds in K-space)
    ax = axes[1, 2]
    # Compare meningiom slice with nearby slices to find anomalous freqs
    if men_idx > 0 and men_idx < len(t1c_images) - 1:
        kspace_prev = compute_kspace(t1c_images[men_idx - 1])
        kspace_next = compute_kspace(t1c_images[men_idx + 1])
        expected = (np.abs(kspace_prev) + np.abs(kspace_next)) / 2
        anomaly = np.abs(mag[:min_r, :min_c] - expected[:min_r, :min_c])
        im = ax.imshow(np.log(anomaly + 1), cmap='hot')
        ax.set_title('Spectral Anomaly Map\n|Actual - Expected|', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'meningiom_kspace_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[SAVED] meningiom_kspace_heatmap.png")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    
    # Paths to series
    t1_path = os.path.join(base, '401c5659', 'f2666859', '1057d9cb')   # Sag T1
    t1c_path = os.path.join(base, '401c5659', 'f2666859', '1057d9eb')  # Sag T1+C
    
    output_dir = os.path.join(base, 'results_meningiom')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  MENINGIOM K-SPACE FOCUSED ANALYSIS")
    print("  Frontal Meningiom - Sag T1 vs T1+C")
    print("=" * 60)
    
    print("\nLoading Sag T1 (pre-contrast)...")
    t1_images = load_series_from_dir(t1_path)
    print(f"  [OK] {len(t1_images)} slices")
    
    print("Loading Sag T1+C (post-contrast)...")
    t1c_images = load_series_from_dir(t1c_path)
    print(f"  [OK] {len(t1c_images)} slices")
    
    meningiom_slice = 11  # Highest dK score location
    
    print(f"\nMeningiom focus: Slice {meningiom_slice}")
    print(f"Region of interest: Slices 9-13")
    
    print("\n[1/3] Generating K-Space Filtering Analysis...")
    plot_meningiom_filtering(t1_images, t1c_images, meningiom_slice, output_dir)
    
    print("[2/3] Generating Contrast Comparison...")
    plot_contrast_comparison(t1_images, t1c_images, output_dir)
    
    print("[3/3] Generating K-Space Heatmap...")
    plot_meningiom_heatmap(t1c_images, output_dir)
    
    print("\n[DONE] Meningiom analysis complete!")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
