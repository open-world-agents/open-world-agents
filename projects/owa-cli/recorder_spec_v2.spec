# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks.gi import get_gi_typelibs
import os

# Initialize lists to collect binaries, datas, and hiddenimports
binaries = []
datas = []
hiddenimports = [
    'gi',
    'gi.overrides',
    'gi._gi',
    'gi._option',
    'gi.repository',
    'gi.repository.GObject',
    'gi.repository.GLib',
    'gi.repository.Gio',
    # Any other base gi modules you need
]

# List of GStreamer typelibs you need
gstreamer_typelibs = [
    ('Gst', '1.0'),
    ('GstBase', '1.0'),
    ('GstVideo', '1.0'),
    ('GstAudio', '1.0'),
    ('GstApp', '1.0'),
    ('GstPbutils', '1.0'),
    # Add other typelibs if needed
]

for typelib_name, typelib_version in gstreamer_typelibs:
    bin, dat, himports = get_gi_typelibs(typelib_name, typelib_version)
    binaries += bin
    datas += dat
    hiddenimports += himports

gst_binaries = [
    (os.path.join(r'C:\Users\MilkClouds\miniforge3\pkgs\gstreamer-1.24.6-h5006eae_0\Library\bin', 'gst-launch-1.0.exe'), '.'),  # Include in root directory
    (os.path.join(r'C:\Users\MilkClouds\miniforge3\pkgs\gstreamer-1.24.6-h5006eae_0\Library\bin', 'gst-inspect-1.0.exe'), '.'),
]

datas += gst_binaries

# Include any additional hidden imports from other packages
additional_hiddenimports = [
    'owa',
    'owa_env_desktop',
    'owa_env_gst',
    # Add any hidden imports used by these packages
]

hiddenimports += additional_hiddenimports

a = Analysis(
    ['recorder.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        "gstreamer": {}
    },
    runtime_hooks=[],
    excludes=[],
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
    name='recorder',
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
