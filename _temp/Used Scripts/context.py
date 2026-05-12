with open("pdf_excerpt.txt", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.split('\n')
for i, line in enumerate(lines):
    if 'mean value of 4.89 m' in line:
        for j in range(max(0, i-5), min(len(lines), i+6)):
            print(lines[j])
