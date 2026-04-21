import traceback
import os

pdf_path = r"c:\Users\saisi\OneDrive\Desktop\SURA\SURA\Research Papers (Previous)\MagWi_Benchmark_Dataset_for_Long_Term_Magnetic_Field_and_Wi-Fi_Data_Involving_Heterogeneous_Smartphones_Multiple_Orientations_Spatial_Diversity_and_Multi-Floor_Buildings.pdf"

try:
    import PyPDF2
    with open("pdf_excerpt.txt", "w", encoding="utf-8") as f:
        reader = PyPDF2.PdfReader(pdf_path)
        for page in reader.pages:
            f.write(page.extract_text() + "\n")
    print("Successfully extracted PDF text using PyPDF2")
except Exception as e:
    print("Failed with PyPDF2. Trying fitz...")
    try:
        import fitz
        with open("pdf_excerpt.txt", "w", encoding="utf-8") as f:
            doc = fitz.open(pdf_path)
            for page in doc:
                f.write(page.get_text())
        print("Successfully extracted PDF text using fitz")
    except Exception as e:
        print("Failed with fitz. Please pip install PyPDF2 or PyMuPDF.")
        import subprocess
        subprocess.run(["pip", "install", "PyPDF2", "--quiet"])
        import PyPDF2
        with open("pdf_excerpt.txt", "w", encoding="utf-8") as f:
            reader = PyPDF2.PdfReader(pdf_path)
            for page in reader.pages:
                f.write(page.extract_text() + "\n")
        print("Successfully installed and extracted PDF text using PyPDF2")
