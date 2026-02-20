import numpy as np
import math

def extract_radial_features(kspace_mag, num_bins=8):
    """
    Extracts statistical features from concentric rings in K-Space.
    This simulates a 'Virtual Biopsy' looking for anomalous energy distributions.
    """
    rows, cols = kspace_mag.shape
    center_r, center_c = rows // 2, cols // 2
    
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - center_c)**2 + (Y - center_r)**2)
    max_dist = np.max(dist_from_center)
    
    # Define bin edges
    bin_edges = np.linspace(0, max_dist, num_bins + 1)
    
    features = []
    
    for i in range(num_bins):
        # Create mask for current ring
        mask = (dist_from_center >= bin_edges[i]) & (dist_from_center < bin_edges[i+1])
        ring_data = kspace_mag[mask]
        
        if len(ring_data) > 0:
            # We use log1p for stability as DC component can be huge
            energy = np.mean(np.log1p(ring_data))
            variance = np.var(np.log1p(ring_data))
            features.append({
                "bin": i+1,
                "radius_range": (float(bin_edges[i]), float(bin_edges[i+1])),
                "energy": float(energy),
                "variance": float(variance)
            })
        else:
            features.append({
                "bin": i+1,
                "radius_range": (float(bin_edges[i]), float(bin_edges[i+1])),
                "energy": 0.0,
                "variance": 0.0
            })
            
    return features


def calculate_angular_entropy(kspace_mag, num_sectors=8):
    """
    Calculates entropy in angular sectors (pie slices) of K-Space.
    Asymmetric tumors (like meningiomas) often disrupt angular frequency symmetry.
    """
    rows, cols = kspace_mag.shape
    center_r, center_c = rows // 2, cols // 2
    
    Y, X = np.ogrid[:rows, :cols]
    # Calculate angle from -pi to pi
    angles = np.arctan2(Y - center_r, X - center_c)
    
    sector_edges = np.linspace(-np.pi, np.pi, num_sectors + 1)
    
    sectors = []
    for i in range(num_sectors):
        mask = (angles >= sector_edges[i]) & (angles < sector_edges[i+1])
        sector_data = kspace_mag[mask]
        
        if len(sector_data) > 0:
            energy = np.sum(np.log1p(sector_data))
            sectors.append(float(energy))
        else:
            sectors.append(0.0)
            
    # Calculate entropy of energy distribution
    total_energy = sum(sectors)
    if total_energy == 0:
        return 0.0
        
    probs = [s / total_energy for s in sectors if s > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return float(entropy)


def generate_kspace_fingerprint(kspace_mag, kspace_phase):
    """
    Generates a high-dimensional vector 'fingerprint' of the K-Space slice.
    Combines Magnitude Radial Energy, Phase Variance, and Angular Entropy.
    """
    radial = extract_radial_features(kspace_mag, num_bins=8)
    radial_energies = [f['energy'] for f in radial]
    
    angular_entropy = calculate_angular_entropy(kspace_mag, num_sectors=8)
    
    # Phase complexity (std dev of phase angles)
    phase_complexity = float(np.std(kspace_phase))
    
    # The fingerprint vector (Length 10)
    fingerprint = radial_energies + [angular_entropy, phase_complexity]
    
    # Optional: Normalize the vector
    norm = np.linalg.norm(fingerprint)
    if norm > 0:
        fingerprint = (np.array(fingerprint) / norm).tolist()
        
    return fingerprint


def similarity_score(fp1, fp2):
    """
    Cosine similarity between two K-Space fingerprints (0 to 1).
    """
    fp1 = np.array(fp1)
    fp2 = np.array(fp2)
    
    dot = np.dot(fp1, fp2)
    norm1 = np.linalg.norm(fp1)
    norm2 = np.linalg.norm(fp2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(dot / (norm1 * norm2))


def image_free_pathology_detection(fingerprints, baseline_fingerprint):
    """
    Simulates a 1D CNN / Similarity model detecting anomalies without reconstructing the Image.
    Returns anomaly probability per slice.
    """
    detection_results = []
    for i, fp in enumerate(fingerprints):
        # Calculate similarity to baseline (e.g., normal brain tissue signature)
        sim = similarity_score(fp, baseline_fingerprint)
        
        # If similarity drops below threshold, we suspect pathology
        # Convert similarity to an 'anomaly score' (0 = normal, 1 = highly anomalous)
        anomaly_score = 1.0 - sim
        
        # Non-linear scaling to make anomalies 'pop'
        anomaly_prob = min(1.0, math.pow(anomaly_score * 5.0, 2))
        
        detection_results.append({
            "slice_index": i,
            "similarity": float(sim),
            "anomaly_probability": float(anomaly_prob),
            "flagged": bool(anomaly_prob > 0.5)
        })
        
    return detection_results
