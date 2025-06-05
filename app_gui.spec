# app_gui.spec  ——  放在项目根目录

from PyInstaller.utils.hooks import collect_submodules
import os, sys

block_cipher = None

# ===== 1) 把 DLL “搬” 进来 ====================================================
ANACONDA_DLL = r"D:\software\Anaconda\Library\bin"
mkldlls = [
    "mkl_intel_thread.2.dll",
    "mkl_core.2.dll",
    "mkl_avx2.2.dll",
    "mkl_def.2.dll",
    "libiomp5md.dll",
]
binaries = [(os.path.join(ANACONDA_DLL, d), ".") for d in mkldlls]

# ===== 2) 需要一并分发的权重 / 字体 ==========================================
datas = [
    # 整个 MTCNN 权重目录
    ("weights/facenet_inception_resnetv1.pt", "weights"),
    # YOLO 权重
    ("yolo/weights/best202505203.pt", "yolo/weights"),
    ("yolo/weights/yolov8n.pt", "yolo/weights"),
    # 字体
    ("bgtx.ttf", "."),
]

a = Analysis(
    ["app_gui.py"],           # 入口脚本
    pathex=["."],             # 搜索路径
    binaries=binaries,
    datas=datas,
    hiddenimports=collect_submodules("torch")
                  + collect_submodules("PIL"),   # 常见缺漏
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=False,
    name="main",          # 生成 main.exe
    console=True,         # 先开控制台调试；确认 OK 再改 False
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="main",
)
