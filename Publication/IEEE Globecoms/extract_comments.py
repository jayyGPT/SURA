import sys
sys.stdout.reconfigure(encoding='utf-8')
import fitz

doc = fitz.open(r'c:\Users\lenovo\Documents\GitHub\SURA\Publication\IEEE Globecoms\FeedBack from Professor.pdf')

# Only extract blue text (professor comments) with surrounding context
for i, page in enumerate(doc):
    blocks = page.get_text('dict')['blocks']
    for b in blocks:
        if b['type'] == 0:
            for line in b['lines']:
                parts = []
                has_blue = False
                for span in line['spans']:
                    c = span['color']
                    r = (c >> 16) & 0xFF
                    g = (c >> 8) & 0xFF
                    bv = c & 0xFF
                    txt = span['text']
                    if r == 0 and g == 0 and bv == 255:  # pure blue
                        parts.append(f'>>>{txt}<<<')
                        has_blue = True
                    else:
                        parts.append(txt)
                if has_blue:
                    full_line = ''.join(parts)
                    print(f'P{i+1}: {full_line}')

doc.close()
