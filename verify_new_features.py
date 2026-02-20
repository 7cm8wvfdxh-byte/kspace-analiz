import requests
import os
import time
import shutil

BASE_URL = "http://localhost:8000"
DICOM_DIR = r"c:\Users\TUGRUL\.gemini\antigravity\scratch\dicom_kspace\401c5659\f2666859\1057d9eb"

def get_studies():
    try:
        return requests.get(f"{BASE_URL}/api/studies").json()
    except:
        return []

def delete_study(study_id):
    print(f"Deleting study {study_id}...")
    requests.delete(f"{BASE_URL}/api/study/{study_id}")

def upload_study(suffix=""):
    print(f"Uploading study {suffix}...")
    files = []
    # Limit to 5 files for speed
    count = 0
    for fname in os.listdir(DICOM_DIR):
        if count >= 5: break
        fpath = os.path.join(DICOM_DIR, fname)
        if os.path.isfile(fpath):
            # Hack: Rename slightly to simulate different file (if server checks filename)
            # Actually server uses filename in multipart data.
            files.append(('files', (f"{suffix}_{fname}", open(fpath, 'rb'), 'application/dicom')))
            count += 1
    
    r = requests.post(f"{BASE_URL}/api/upload", files=files)
    if r.status_code != 200:
        print(f"Upload failed: {r.text}")
        return None
    return r.json()['study_id']

def main():
    print("--- Verification Start ---")
    
    # Clean up existing studies to be safe (optional, or just use new ones)
    studies = get_studies()
    for s in studies:
        # Delete if it has 0 files or known bad
        if s.get('file_count', 0) == 0 or 'zip' in str(s).lower():
            delete_study(s['id'])

    # Create two new studies
    id1 = upload_study("A")
    time.sleep(1)
    id2 = upload_study("B")
    
    if not id1 or not id2:
        print("Failed to create studies.")
        return

    print(f"Studies created: {id1}, {id2}")
    
    # 1. Test Comparison
    print(f"Comparing {id1} vs {id2}...")
    try:
        r = requests.post(f"{BASE_URL}/api/compare", 
                          json={"study_id_1": id1, "study_id_2": id2})
        if r.status_code == 200:
            res = r.json()
            print("[PASS] Comparison API")
            print(f"  Plot URL: {res.get('plot_url')}")
            print(f"  Metrics count: {len(res.get('metrics', []))}")
        else:
            print(f"[FAIL] Comparison API: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"[FAIL] Comparison API exception: {e}")

    # 2. Test 3D Volume
    print(f"Fetching 3D volume for {id1}...")
    try:
        r = requests.get(f"{BASE_URL}/api/volume/{id1}")
        if r.status_code == 200:
            res = r.json()
            print("[PASS] Volume API")
            print(f"  Points count: {res.get('count')}")
        else:
            print(f"[FAIL] Volume API: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"[FAIL] Volume API exception: {e}")

if __name__ == "__main__":
    main()
