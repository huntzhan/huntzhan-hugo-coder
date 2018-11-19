import requests

# API_URL = 'https://api.github.com/repos/gohugoio/hugo/releases/latest'
# data = requests.get(API_URL).json()
#
# from pprint import pprint
# pprint(data)
#
# item = list(filter(
#     lambda obj: obj['name'].endswith('Linux-64bit.deb') and 'extended' not in obj['name'],
#     data['assets'],
# ))
# assert len(item) == 1
# browser_download_url = item[0]['browser_download_url']

# Investigating.
#
# Hardcode to mitigate the API rate limiting error.
# {u'documentation_url': u'https://developer.github.com/v3/#rate-limiting',
#  u'message': u"API rate limit exceeded for 35.192.136.167. (But here's the good news: Authenticated requests get a higher rate limit. Check out the documentation for more details.)"}
#
browser_download_url = 'https://github.com/gohugoio/hugo/releases/download/v0.51/hugo_0.51_Linux-64bit.deb'

with open('hugo.deb', 'wb') as fout:
    print(browser_download_url)
    rep = requests.get(browser_download_url)
    fout.write(rep.content)
