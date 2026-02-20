import requests
import time

base_url = "http://localhost:8000"

def main():
    try:
        # List studies
        print("Fetching studies...")
        r = requests.get(f"{base_url}/api/studies")
        studies = r.json()
        print(f"Found {len(studies)} studies.")
        
        if not studies:
            print("No studies found. Upload might have failed.")
            return

        # Get latest study
        latest = studies[0]
        study_id = latest['id']
        status = latest['status']
        print(f"Latest Study: {study_id} | Status: {status}")
        
        if status == 'uploaded':
            print("Triggering analysis...")
            r = requests.post(f"{base_url}/api/analyze/{study_id}")
            print(f"Analyze response: {r.json()}")
            
        # Poll
        while True:
            r = requests.get(f"{base_url}/api/study/{study_id}")
            study = r.json()
            status = study['status']
            print(f"Status: {status}")
            
            if status == 'completed':
                print("Analysis COMPLETE!")
                print(f"Results summary: {study.get('results', {}).get('summary')}")
                break
            if status == 'error':
                print(f"Analysis ERROR: {study.get('error')}")
                break
            
            time.sleep(2)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
