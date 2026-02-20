import requests
import os
import time

# DICOM directory
dicom_dir = r"c:\Users\TUGRUL\.gemini\antigravity\scratch\dicom_kspace\401c5659\f2666859\1057d9eb"
url = "http://localhost:8000/api/upload"

files_to_upload = []
for fname in os.listdir(dicom_dir):
    fpath = os.path.join(dicom_dir, fname)
    if os.path.isfile(fpath):
        files_to_upload.append(('files', (fname, open(fpath, 'rb'), 'application/dicom')))

print(f"Uploading {len(files_to_upload)} files...")
try:
    response = requests.post(url, files=files_to_upload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    study_id = response.json().get('study_id')
    if study_id:
        print(f"Study ID: {study_id}")
        
        # Trigger analysis (although UI does it automatically, API might not unless configured)
        # The UI logic is: upload -> get study_id -> call /api/analyze/{study_id}
        # Let's call analyze manually
        analyze_url = f"http://localhost:8000/api/analyze/{study_id}"
        print(f"Triggering analysis at {analyze_url}")
        requests.post(analyze_url)
        
        # Poll for completion
        status_url = f"http://localhost:8000/api/study/{study_id}"
        while True:
            r = requests.get(status_url)
            status = r.json().get('status')
            print(f"Status: {status}")
            if status in ['completed', 'error']:
                break
            time.sleep(1)
            
except Exception as e:
    print(f"Error: {e}")
