# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Automatically collect all non-Python data files from ultralytics, super_gradients, urllib3, and botocore
ultralytics_datas = collect_data_files('ultralytics')
super_gradients_datas = collect_data_files('super_gradients')
urllib3_datas = collect_data_files('urllib3')
botocore_datas = collect_data_files('botocore')

# Combine the data from all packages
datas = ultralytics_datas + super_gradients_datas + urllib3_datas + botocore_datas

# Collect all hidden submodules from super_gradients and botocore, including urllib3 and ssl_
hiddenimports = collect_submodules('super_gradients') + collect_submodules('botocore') + [
    'matplotlib.backends.backend_agg',
    'ssl',  # Ensure ssl is bundled
    'urllib3.util.ssl_',  # Ensure urllib3 SSL utility is bundled
    'pycocotools',
]

# Add custom hook for urllib3 (optional)
hookspath=['.']  # Add the directory where your custom hook for urllib3 is located

# Set your project base directory (dynamically determined path)
project_dir = os.path.abspath('.')

a = Analysis(
    ['run_local.py'],
    pathex=[project_dir, os.path.join(project_dir, 'backend')],
    binaries=[],
    datas=datas,  # Include the collected data files
    hiddenimports=hiddenimports,  # Include hidden imports
    hookspath=hookspath,  # Include hooks path for custom hooks
    runtime_hooks=[],
    excludes=['super_gradients.modules.quantization'],  # Exclude quantization module if needed
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='run_local',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

