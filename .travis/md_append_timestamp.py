import sys
import re

md_path = sys.argv[1]
print('md_path', md_path)

with open(md_path) as fin:
    lines = fin.readlines()

processed = []
for line in lines:
    line = line.rstrip()

    if not ''.join(line.strip().split()).startswith('*[x]'):
        processed.append(line)
        continue

    pat = r'\s*@\((\d{4}-\d{2}-\d{2} \d{2}:\d{2})\)\s*'
    match = re.search(pat, line)
    assert match
    
    dt_prefix = match.group(1)[:len('yyyy-mm-dd')]   
    suffix = f'<div style="display:inline; float:right; font-family:serif;"><strong>Done {dt_prefix}</strong></div>'
    
    processed.append(re.sub(pat, ' ', line) + suffix)
        
out_path = sys.argv[2]
print('out_path', out_path)

with open(out_path, 'w') as fout:
    fout.write('\n'.join(processed))