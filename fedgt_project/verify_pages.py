import urllib.request, urllib.parse, http.cookiejar, re

jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

# Get CSRF token from login page
resp = opener.open('http://127.0.0.1:8000/login/')
html = resp.read().decode()
m = re.search(r'csrfmiddlewaretoken.*?value=["\']([^"\']+)', html)
csrf = m.group(1) if m else ''
print('CSRF found:', bool(csrf))

# Login
data = urllib.parse.urlencode({
    'username': 'admin', 'password': 'admin',
    'csrfmiddlewaretoken': csrf
}).encode()
req = urllib.request.Request(
    'http://127.0.0.1:8000/login/', data=data,
    headers={'Referer': 'http://127.0.0.1:8000/login/'}
)
resp = opener.open(req)
print('After login URL:', resp.url)

# Test /fedgt/
resp = opener.open('http://127.0.0.1:8000/fedgt/')
print('FedGT page status:', resp.status)
body = resp.read().decode()
print('Has FedGT Engine text   :', 'FedGT Engine' in body)
print('Has Run FedGT Decode    :', 'Run FedGT Decode' in body)
print('Has Aggregate All Rounds:', 'Aggregate All Rounds' in body)
print('Page size               :', len(body), 'bytes')

# Test /clients/
resp2 = opener.open('http://127.0.0.1:8000/clients/')
print('\nClients page status:', resp2.status)
body2 = resp2.read().decode()
print('Has Client Monitor text :', 'Client Monitor' in body2)

# Test /training/
resp3 = opener.open('http://127.0.0.1:8000/training/')
print('\nTraining page status:', resp3.status)
body3 = resp3.read().decode()
print('Has Secure Aggregation  :', 'Secure Aggregation' in body3)

print('\nAll pages OK!')
