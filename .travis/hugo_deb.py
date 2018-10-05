import requests

API_URL = 'https://api.github.com/repos/gohugoio/hugo/releases/latest'
data = requests.get(API_URL).json()

item = list(filter(
    lambda obj: obj['name'].endswith('Linux-64bit.deb') and 'extended' not in obj['name'],
    data['assets'],
))
assert len(item) == 1
browser_download_url = item[0]['browser_download_url']

with open('hugo.deb', 'wb') as fout:
    rep = requests.get(browser_download_url)
    fout.write(rep.content)
