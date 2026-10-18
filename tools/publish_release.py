"""Publish the already-validated v0.5.0 assets using existing Git credentials.

No credentials are written to disk or printed. Requires an already-pushed tag.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO='joeboy0722/vonnx'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish',action='store_true',help='Create draft, upload verified assets, publish')
    args=parser.parse_args()
    credential=subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','credential','fill'],
        input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,check=True,
        env={**os.environ,'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'never'})
    fields=dict(line.split('=',1) for line in credential.stdout.splitlines() if '=' in line)
    token=fields.get('password')
    if not token:raise RuntimeError('GitHub credential unavailable')

    def request(path,method='GET',data=None,content_type='application/json'):
        url=path if path.startswith('https://') else 'https://api.github.com/repos/'+REPO+path
        if urllib.parse.urlparse(url).hostname not in ('api.github.com','uploads.github.com'):
            raise ValueError('Unexpected GitHub endpoint')
        if isinstance(data,dict):data=json.dumps(data).encode('utf-8')
        req=urllib.request.Request(url,method=method,data=data,headers={
            'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
            'Content-Type':content_type,'User-Agent':'vONNX-release'})
        with urllib.request.urlopen(req,timeout=180) as response:
            return json.load(response)

    repo=request('')
    print('Repository:',repo['full_name'],'push permission:',repo.get('permissions',{}).get('push'))
    if not args.publish:return
    version='0.5.0'; tag='v'+version
    ap=json.loads((ROOT/'docs/ap_v050.json').read_text(encoding='utf-8'))
    binary=next((ROOT/'vonnx').glob('vonnx_py*.pyd'))
    if not ap['passed'] or hashlib.sha256(binary.read_bytes()).hexdigest()!=ap['binary_sha256']:
        raise RuntimeError('AP gate does not identify this binary')
    files=[ROOT/'dist'/name for name in (f'vonnx-{version}-cp312-cp312-win_amd64.whl',
        f'vonnx-{version}-windows-x64-sdk.zip','USAGE.md','AP_RESULTS.json','SHA256SUMS.txt')]
    sums=(ROOT/'dist/SHA256SUMS.txt').read_text(encoding='utf-8')
    for file in files[:-1]:
        if f'{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.name}' not in sums:
            raise RuntimeError('Asset checksum mismatch: '+file.name)
    try:
        existing=request('/releases/tags/'+tag)
    except urllib.error.HTTPError as exc:
        if exc.code!=404:raise
        existing=None
    if existing:raise RuntimeError('Release already exists; refusing to replace it')
    request('/git/ref/tags/'+tag)
    release=request('/releases',method='POST',data=dict(tag_name=tag,name='vONNX '+tag,
        body=(ROOT/'docs/release_notes_v050.md').read_text(encoding='utf-8'),draft=True,prerelease=False))
    upload=release['upload_url'].split('{')[0]
    for file in files:
        asset=request(upload+'?name='+urllib.parse.quote(file.name),method='POST',data=file.read_bytes(),
                      content_type='application/octet-stream')
        if asset['size']!=file.stat().st_size:raise RuntimeError('Asset size mismatch')
        print('Uploaded:',asset['name'],asset['size'])
    published=request('/releases/'+str(release['id']),method='PATCH',data=dict(draft=False,make_latest='true'))
    verified=request('/releases/tags/'+tag)
    if verified['draft'] or {a['name'] for a in verified['assets']}!={f.name for f in files}:
        raise RuntimeError('Published release verification failed')
    print('Published:',published['html_url'])


if __name__=='__main__':main()
