# -*- mode: python ; coding: utf-8 -*-
"""
Reusable PyInstaller spec for PointLabeler (PySide6 + pyvista + pyvistaqt + vtk).

Usage:
  pyinstaller --noconfirm --clean PointLabeler.spec
"""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


# In PyInstaller spec execution context, __file__ may be undefined.
# SPECPATH is provided by PyInstaller and points to the spec directory.
project_root = Path(globals().get("SPECPATH", Path.cwd())).resolve()
src_root = project_root / "src"
entry_script = src_root / "point_labeler" / "app" / "main.py"
if not entry_script.exists():
    raise FileNotFoundError(f"Entry script not found: {entry_script}")


# Collect package data/binaries/hiddenimports for VTK and PyVista stack.
datas = []
binaries = []
hiddenimports = []

# Avoid collecting optional pyvista.trame stack (requires extra deps and is
# irrelevant for this desktop app).
datas += collect_data_files("pyvista")
binaries += collect_dynamic_libs("pyvista")
hiddenimports += [
    m for m in collect_submodules("pyvista")
    if not (m == "pyvista.trame" or m.startswith("pyvista.trame."))
]

for pkg in ("pyvistaqt", "vtkmodules"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Ensure point_labeler package is discoverable when running from frozen app.
pathex = [str(src_root)]


a = Analysis(
    [str(entry_script)],
    pathex=pathex,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pyvista.trame"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PointLabeler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PointLabeler",
)
