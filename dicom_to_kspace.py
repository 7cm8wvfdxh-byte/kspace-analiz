import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
import sys
import imageio
import glob

def generate_phantom(size=256):
    """
    Generates a simple synthetic phantom (image) for demonstration.
    Returns a 2D numpy array.
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create a circle
    radius = 0.5
    phantom = (X**2 + Y**2) < radius**2
    
    # Add a smaller rectangle (simulating a "lesion" or structure)
    phantom = phantom.astype(float)
    phantom[(X > 0.1) & (X < 0.3) & (Y > 0.1) & (Y < 0.3)] = 2.0
    
    # Add some gaussian noise
    noise = np.random.normal(0, 0.1, size=(size, size))
    phantom += noise
    
    return [phantom] # Return as a list of one image

def load_dicom_series_from_dir(path):
    """
    Handles loading from a DICOMDIR or a regular directory.
    Returns a list of pixel arrays sorted by Instance Number.
    """
    images = []
    
    if os.path.basename(path) == "DICOMDIR":
        print("Detected DICOMDIR file. Reading directory...")
        try:
            ds = pydicom.dcmread(path)
            # Iterate through records to find IMAGEs
            # We'll collect (instance_number, pixel_array) tuples to sort later
            image_records = []
            
            base_dir = os.path.dirname(path)
            
            for record in ds.DirectoryRecordSequence:
                if record.DirectoryRecordType == "IMAGE":
                    # Construct full path to the image file
                    image_parts = record.ReferencedFileID
                    if isinstance(image_parts, str):
                        image_parts = [image_parts]
                    
                    image_path = os.path.join(base_dir, *image_parts)
                    
                    try:
                        dcm = pydicom.dcmread(image_path)
                        # Get Instance Number for sorting, default to 0 if missing
                        instance_num = int(dcm.get('InstanceNumber', 0))
                        image_records.append((instance_num, dcm.pixel_array))
                    except Exception as e:
                        print(f"Failed to load {image_path}: {e}")
            
            # Sort by instance number
            image_records.sort(key=lambda x: x[0])
            images = [img for _, img in image_records]
            
            print(f"Total images loaded: {len(images)}")
            return images
            
        except Exception as e:
            print(f"Error reading DICOMDIR: {e}")
            return []
    elif os.path.isdir(path):
        # Allow loading from a directory of dcm files directly (without DICOMDIR)
        print(f"Scanning directory: {path}")
        image_records = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".dcm") or "." not in file: # strict check might miss some
                    full_path = os.path.join(root, file)
                    try:
                        dcm = pydicom.dcmread(full_path, stop_before_pixels=False)
                        if 'PixelData' in dcm:
                             instance_num = int(dcm.get('InstanceNumber', 0))
                             image_records.append((instance_num, dcm.pixel_array))
                    except:
                        pass
        
        image_records.sort(key=lambda x: x[0])
        images = [img for _, img in image_records]
        print(f"Total images loaded from directory: {len(images)}")
        return images

    else:
        # Single file load
        img = load_dicom(path)
        return [img] if img is not None else []

def load_dicom(filepath):
    """
    Loads a DICOM file and returns the pixel array.
    """
    try:
        ds = pydicom.dcmread(filepath)
        return ds.pixel_array
    except Exception as e:
        print(f"Error loading DICOM file: {e}")
        return None

def compute_kspace(image_data):
    """
    Computes the K-space (frequency domain) representation of an image
    using 2D Inverse Fast Fourier Transform (iFFT).
    """
    kspace = np.fft.fft2(image_data)
    kspace_shifted = np.fft.fftshift(kspace)
    return kspace_shifted

def apply_filter(kspace_data, filter_type='lowpass', radius=30):
    """
    Applies a filter mask to the K-space data.
    radius: radius of the filter in pixels (from center)
    """
    rows, cols = kspace_data.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create a mask grid
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # Distance from center
    dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    
    mask = np.zeros((rows, cols))
    
    if filter_type == 'lowpass':
        # Keep center frequencies (contract, structure)
        mask[dist <= radius] = 1
    elif filter_type == 'highpass':
        # Keep peripheral frequencies (edges, noise)
        mask[dist > radius] = 1
    elif filter_type == 'notch':
        # Remove a specific band (experimental)
        mask[:] = 1
        mask[(dist > radius - 5) & (dist < radius + 5)] = 0
        
    # Apply mask
    filtered_kspace = kspace_data * mask
    return filtered_kspace, mask

def reconstruct_image(kspace_data):
    """
    Reconstructs the image from K-space data using Inverse FFT.
    """
    # Shift back (if strictly following FFT/iFFT conventions, standard numpy fft2 expects 0-freq at corner)
    # But since we shifted 0-freq to center for visualization and filtering, we must shift back before iFFT.
    kspace_unshifted = np.fft.ifftshift(kspace_data)
    
    # Inverse FFT
    # Note: If we used fft2 to go forward, we use ifft2 to go back.
    img_reconstructed = np.fft.ifft2(kspace_unshifted)
    
    # Return magnitude (since original image was real, reconstruction might have small imaginary parts)
    return np.abs(img_reconstructed)

def visualize_filtering(image_data, radius=30):
    """
    Visualizes original image, K-space, Filter Mask, and Reconstructed Image.
    """
    kspace = compute_kspace(image_data)
    
    # Apply Filters
    kspace_low, mask_low = apply_filter(kspace, 'lowpass', radius)
    img_low = reconstruct_image(kspace_low)
    
    kspace_high, mask_high = apply_filter(kspace, 'highpass', radius)
    img_high = reconstruct_image(kspace_high)
    
    plt.figure(figsize=(15, 10))
    
    # 1. Original
    plt.subplot(2, 3, 1)
    plt.imshow(image_data, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(np.log(np.abs(kspace) + 1), cmap='gray')
    plt.title("Original K-Space")
    plt.axis('off')

    # 2. Low Pass (Center only)
    plt.subplot(2, 3, 2)
    plt.imshow(img_low, cmap='gray')
    plt.title(f"Low Pass Filter (Radius={radius})\n(Contrast/Structure)")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(np.log(np.abs(kspace_low) + 1), cmap='gray')
    plt.title("Masked K-Space (Low Pass)")
    plt.axis('off')

    # 3. High Pass (Periphery only)
    plt.subplot(2, 3, 3)
    plt.imshow(img_high, cmap='gray')
    plt.title(f"High Pass Filter (Radius={radius})\n(Edges/Details)")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(np.log(np.abs(kspace_high) + 1), cmap='gray')
    plt.title("Masked K-Space (High Pass)")
    plt.axis('off')
    
    plt.tight_layout()
    output_file = 'kspace_filtering_analysis.png'
    plt.savefig(output_file)
    print(f"Analysis saved to {output_file}")
    plt.show()

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        # Just grab the first valid image for filtering analysis
        images = load_dicom_series_from_dir(path)
        if images:
            # Use middle slice for better anatomy visibility
            image_data = images[len(images)//2]
        else:
            return
    else:
        print("No DICOM file/dir provided. Generating synthetic phantom...")
        image_data = generate_phantom()[0]

    # Analyze filtering with default radius
    print("Performing K-space filtering analysis...")
    visualize_filtering(image_data, radius=30)

if __name__ == "__main__":
    main()
