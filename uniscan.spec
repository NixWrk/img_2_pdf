# PyInstaller build definition for the portable Windows distribution.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


datas = collect_data_files(
    "tkinterdnd2",
    includes=[
        "tkdnd/win-x64/*.dll",
        "tkdnd/win-x64/*.tcl",
        "tkdnd/win-x64-tcl9/*.dll",
        "tkdnd/win-x64-tcl9/*.tcl",
    ],
)
hiddenimports = collect_submodules("tkinterdnd2")

a = Analysis(
    ["src/uniscan/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["scripts/pyinstaller_hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["fitz", "onnxruntime", "pymupdf"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="uniscan",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="uniscan",
)
