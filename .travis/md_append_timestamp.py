import sys
from git import Repo

repo = Repo(sys.argv[1])
print('repo', repo)

md_path = sys.argv[2]
print('md_path', md_path)

ignored_date_str = sys.argv[4]

processed = []
for commit, lines in repo.blame('HEAD', md_path):
    date_str = commit.committed_datetime.strftime('%Y-%m-%d')
    if date_str == ignored_date_str:
        date_str = 'INIT'

    suffix = f' &nbsp; &nbsp; \[**Done {date_str}**\]'
    for line in lines:
        if date_str and ''.join(line.strip().split()).startswith('-[x]'):
            line += suffix
        processed.append(line)

out_path = sys.argv[3]
print('out_path', out_path)

with open(out_path, 'w') as fout:
    fout.write('\n'.join(processed))