import jwt
import time
import requests

# JWT を作成（有効期限5分）
with open('C:/Users/Blue-/hal-lifemate-ai.2026-04-25.private-key.pem', 'r') as f:
    pem = f.read()

payload = {'iat': int(time.time()), 'exp': int(time.time()) + 300, 'iss': '3498774'}
token = jwt.encode(payload, pem, algorithm='RS256')

# Installation 一覧を取得
r = requests.get('https://api.github.com/app/installations',
    headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'})

for inst in r.json():
    print(f"ID: {inst['id']}, Account: {inst['account']['login']}")
# → lifemate-ai の Installation ID をメモ