# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files
block_cipher = None

a = Analysis(
    ['VideoAnalysisToolBackend-dev/start.py'],
    pathex=[],
    binaries=[],
    datas=[('VideoAnalysisToolBackend-dev/templates', 'templates'), ('VideoAnalysisToolBackend-dev/static', 'static'),
        ('VideoAnalysisToolBackend-dev/backend', 'backend'), ('VideoAnalysisToolBackend-dev/app/models', 'app/models'), ('VideoAnalysisToolBackend-dev/app/analysis/models','app/analysis/models')] + collect_data_files('ultralytics') + collect_data_files('super-gradients'),
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure,a.zipped_data,
             cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VisionMD',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VisionMDTool',
)
