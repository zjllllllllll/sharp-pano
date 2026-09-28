"""Read-only camera-lab server, optionally showing an unpublished revision."""
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit
import argparse
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pano-web'))
import server


class LabHandler(server.Handler):
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/':
            self.send_response(302);self.send_header('Location','/camera-lab');self.end_headers();return
        if path in {'/camera-lab','/camera-lab/','/camera-lab.js','/camera-lab.css','/camera-lab-filter.mjs',
                    '/api/camera-lab','/favicon.svg'} or path.startswith(('/camera-lab-artifacts/','/vendor/three/')):
            return super().do_GET()
        self.send_error(404)

    def do_POST(self):
        self.send_error(405,'This camera lab is read-only')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8769)
    p.add_argument('--run',type=Path,default=HERE/'runs/20260922-adaptive-sampling');a=p.parse_args()
    if a.run.resolve().parent!=(HERE/'runs').resolve():raise ValueError('Invalid run directory')
    server.camera_lab.RUN=a.run.resolve()
    print(f'Camera lab http://127.0.0.1:{a.port}/camera-lab -> {a.run}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),LabHandler).serve_forever()
