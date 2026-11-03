import urllib.request
import json

print("1. Testing Web Page GET / ...")
resp = urllib.request.urlopen("http://localhost:5000/")
html = resp.read().decode("utf-8")
assert "playerOrig" in html and "playerProc" in html
assert "sectionRealtime" not in html
print("   Page OK! (Real-time successfully removed, dedicated players present)")

print("2. Testing Audio Upload /api/record with 44100 Hz user recording...")
boundary = "----WebKitFormBoundaryFinalTest"
with open(r"C:\Users\hemsu\AppData\Local\Temp\autotune_web\06a9b20a36ba.wav", "rb") as f:
    wav_bytes = f.read()

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="audio"; filename="recording.wav"\r\n'
    f"Content-Type: audio/wav\r\n\r\n"
).encode("latin1") + wav_bytes + f"\r\n--{boundary}--\r\n".encode("latin1")

req = urllib.request.Request(
    "http://localhost:5000/api/record",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)
with urllib.request.urlopen(req) as r:
    up = json.loads(r.read().decode())
    file_id = up["file_id"]
    print(f"   Upload OK! File ID: {file_id}, Duration: {up['duration']}s, Sample Rate: {up['sample_rate']} Hz")

print("3. Testing Waveform GET /api/waveform/<file_id> ...")
with urllib.request.urlopen(f"http://localhost:5000/api/waveform/{file_id}") as r:
    wf = json.loads(r.read().decode())
    print(f"   Waveform points: {len(wf['waveform'])}")

print("4. Testing Autotune POST /api/autotune (Tonic: C, Major Scale, Strength: 100%, Transpose: 0) ...")
payload = json.dumps({
    "file_id": file_id,
    "scale": "major",
    "root": "C",
    "strength": 1.0,
    "transpose": 0
}).encode("utf-8")

req = urllib.request.Request(
    "http://localhost:5000/api/autotune",
    data=payload,
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as r:
    res = json.loads(r.read().decode())
    out_id = res["output_id"]
    print(f"   Autotune OK! Output ID: {out_id}, Duration: {res['duration']}s, Peak: {res['peak_output']}")

print("5. Testing Audio Stream GET /api/audio/<output_id> with Range request ...")
req_audio = urllib.request.Request(f"http://localhost:5000/api/audio/{out_id}", headers={"Range": "bytes=0-1000"})
with urllib.request.urlopen(req_audio) as r:
    data = r.read()
    print(f"   Stream status: {r.status}, Bytes read: {len(data)}, Content-Type: {r.headers.get('Content-Type')}")

print("\nALL 5 TESTS PASSED WITH 100% SUCCESS!")
