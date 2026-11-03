import urllib.request

url = "http://localhost:5000/api/audio/bbb300f9f0d9"
req = urllib.request.Request(url, headers={"Range": "bytes=0-1000"})
try:
    with urllib.request.urlopen(req) as r:
        print("Status:", r.status)
        print("Headers:", dict(r.headers))
        data = r.read()
        print("Read bytes:", len(data))
except Exception as e:
    print("Error:", e)
