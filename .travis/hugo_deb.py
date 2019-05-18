import requests

browser_download_url = 'https://github.com/gohugoio/hugo/releases/download/v0.55.5/hugo_0.55.5_Linux-64bit.deb'

with open('hugo.deb', 'wb') as fout:
    print(browser_download_url)
    rep = requests.get(browser_download_url)
    fout.write(rep.content)
