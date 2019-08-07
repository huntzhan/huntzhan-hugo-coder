import requests

browser_download_url = 'https://github.com/gohugoio/hugo/releases/download/v0.56.3/hugo_0.56.3_Linux-64bit.deb'

with open('hugo.deb', 'wb') as fout:
    print(browser_download_url)
    rep = requests.get(browser_download_url)
    fout.write(rep.content)
