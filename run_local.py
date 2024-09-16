import os
import sys
import socket
from django.core.management import execute_from_command_line
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

if __name__ == '__main__':
    # Check if we are in a PyInstaller-built executable
    if getattr(sys, 'frozen', False):
        # Disable Django's autoreload if running inside PyInstaller executable
        settings.DEBUG = False
        os.environ['RUN_MAIN'] = 'true'  # Prevent Django from trying to restart

    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]

    # Print the port number so Electron can read it
    print(port)
    sys.stdout.flush()

    # Start the Django server on the available port, without autoreload
    execute_from_command_line([sys.argv[0], 'runserver', f'127.0.0.1:{port}', '--noreload'])
