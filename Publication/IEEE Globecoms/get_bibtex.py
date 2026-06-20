import urllib.request
import urllib.parse
import json

papers = {
    "magwi": "MagWi: Benchmark dataset for long term magnetic field and Wi-Fi data involving heterogeneous smartphones",
    "radar": "RADAR: An in-building RF-based user location and tracking system",
    "horus": "The Horus WLAN location determination system",
    "deeppos": "DeepPositioning: Intelligent fusion of pervasive magnetic field and WiFi fingerprinting for smartphone indoor localization via deep learning",
    "minloc": "MINLOC: Magnetic field patterns-based indoor localization using convolutional neural networks",
    "kalmannet": "KalmanNet: Neural network aided Kalman filtering for partially known dynamics",
    "wang2024gnn": "Graph Neural Network-Based WiFi Indoor Localization System",
    "rizk2023globloc": "Indoor Localization System for Seamless Tracking Across Buildings and Network Configurations",
    "bilstmmag": "Indoor Localization Using Smartphone Magnetic Sensor Data: A Bi-LSTM Neural Network Approach",
    "driftresistant": "Drift-Resistant Heading Estimation for Smartphone-Based Indoor Positioning via Adaptive Calibration Using Wi-Fi Fingerprinting and Magnetic Stability",
    "axesmapping": "Axes Mapping and Sensor Fusion for Attitude-Unconstrained Pedestrian Dead Reckoning",
    "hybridwifi": "A hybrid indoor positioning solution based on Wi-Fi, magnetic field, and Intertial Navigation",
    "nnwifipdr": "Neural Networks-Based Wi-Fi PDR Indoor Navigation Fusion Methods",
    "miskolc": "Miskolc IIS hybrid IPS: Dataset for hybrid indoor positioning",
    "overviewdatasets": "Empirical Overview of Benchmark Datasets for Geomagnetic Field-Based Indoor Positioning"
}

with open("Ref.bib", "w", encoding="utf-8") as f:
    for key, title in papers.items():
        try:
            url = f"https://api.crossref.org/works?query.title={urllib.parse.quote(title)}&rows=1"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req)
            data = json.loads(response.read().decode('utf-8'))
            
            if data['message']['items']:
                doi = data['message']['items'][0].get('DOI')
                if doi:
                    bib_url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}/transform/application/x-bibtex"
                    bib_req = urllib.request.Request(bib_url, headers={'User-Agent': 'Mozilla/5.0'})
                    bib_response = urllib.request.urlopen(bib_req)
                    bibtex = bib_response.read().decode('utf-8')
                    # Replace the crossref key with our key
                    import re
                    bibtex = re.sub(r'@[a-zA-Z]+\{.*?,', f'@article{{{key},', bibtex, count=1)
                    f.write(bibtex + "\n\n")
                    print(f"Got BibTeX for {key}")
                else:
                    print(f"No DOI for {key}")
            else:
                print(f"No results for {key}")
        except Exception as e:
            print(f"Error for {key}: {e}")
