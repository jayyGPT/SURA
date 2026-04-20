import PyPDF2
import sys

def read_pdf(file_path):
    try:
        reader = PyPDF2.PdfReader(open(file_path, 'rb'))
        
        matches = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_lower = text.lower()
                if any(k in text_lower for k in ['imu', 'particle', 'fusion', 'continuous']):
                    matches.append(f"--- PAGE {i+1} ---\n{text}")
                    
        with open('pdf_extraction.txt', 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(matches))
            
        print(f"Extracted {len(matches)} relevant pages.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    read_pdf('Research Papers (Previous)/MagWi_Benchmark_Dataset_for_Long_Term_Magnetic_Field_and_Wi-Fi_Data_Involving_Heterogeneous_Smartphones_Multiple_Orientations_Spatial_Diversity_and_Multi-Floor_Buildings.pdf')
