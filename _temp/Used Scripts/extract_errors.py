import re
with open("pdf_excerpt.txt", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.split('\n')
with open("errors.txt", "w", encoding="utf-8") as f:
    for line in lines:
        if 'error' in line.lower() or 'accuracy' in line.lower() or 'average' in line.lower() or 'mean' in line.lower():
            f.write(line + "\n")
