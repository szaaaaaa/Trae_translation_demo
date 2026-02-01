import PyInstaller.__main__
import os
import shutil

def build():
    # Clean previous build
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')

    print("Starting build process...")

    PyInstaller.__main__.run([
        'main.py',
        '--name=MeetingTranslator',
        '--console',  # 临时启用控制台以便调试
        '--onedir',
        '--clean',
        '--collect-all=faster_whisper',
        '--collect-all=ctranslate2',
        '--collect-all=tokenizers',
        '--exclude-module=torch',
        '--paths=.', 
    ])

    print("PyInstaller build complete.")

    dist_dir = os.path.join("dist", "MeetingTranslator")
    internal_dir = os.path.join(dist_dir, "_internal")

    # Post-processing: fix OpenMP DLL conflict
    # Only keep ONE copy of libiomp5md.dll in _internal root
    ct2_iomp = os.path.join(internal_dir, "ctranslate2", "libiomp5md.dll")
    root_iomp = os.path.join(internal_dir, "libiomp5md.dll")

    if os.path.exists(ct2_iomp):
        if not os.path.exists(root_iomp):
            print(f"Copying OpenMP runtime to root: {ct2_iomp} -> {root_iomp}")
            shutil.copy2(ct2_iomp, root_iomp)
        # Remove the duplicate to avoid DLL conflicts
        print(f"Removing duplicate OpenMP DLL: {ct2_iomp}")
        os.remove(ct2_iomp)

    print("Build output in dist/MeetingTranslator")

if __name__ == "__main__":
    build()
