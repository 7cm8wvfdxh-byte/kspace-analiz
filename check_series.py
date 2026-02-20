import pydicom
import os

base = r'C:\Users\TUGRUL\.gemini\antigravity\scratch\dicom_kspace\401c5659\f2666859'
dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

for sd in dirs:
    sd_path = os.path.join(base, sd)
    files = os.listdir(sd_path)
    first_file = os.path.join(sd_path, files[0])
    try:
        dcm = pydicom.dcmread(first_file)
        desc = str(dcm.get('SeriesDescription', 'N/A'))
        mod = str(dcm.get('Modality', 'N/A'))
        body = str(dcm.get('BodyPartExamined', 'N/A'))
        rows = dcm.get('Rows', '?')
        cols = dcm.get('Columns', '?')
        print(f"{sd} ({len(files)} files): {desc} | {mod} | {body} | {rows}x{cols}")
    except Exception as e:
        print(f"{sd}: ERROR - {e}")
