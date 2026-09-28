"""Open the existing local workbench, or run it in this terminal."""
import json
import threading
import urllib.error
import urllib.request
import webbrowser
from server import main

URL = 'http://127.0.0.1:8765'


def launch():
    try:
        with urllib.request.urlopen(URL+'/api/state',timeout=2) as response:
            state=json.load(response)
        if 'token' in state and 'health' in state and 'sharp' in state['health']:
            webbrowser.open(URL)
            print('Workbench is already running: '+URL)
            return
    except (OSError, ValueError, urllib.error.URLError):
        pass
    opener=threading.Timer(1.5,lambda:webbrowser.open(URL))
    opener.daemon=True
    opener.start()
    main()


if __name__=='__main__':
    launch()
