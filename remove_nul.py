import os

file_path = r'\\.\D:\Trae_translation\nul'
normal_path = r'D:\Trae_translation\nul'

print(f"Checking {file_path}")
if os.path.exists(file_path):
    print("File exists (accessed via \\\\.\\ prefix)")
    try:
        os.remove(file_path)
        print("Successfully removed file.")
    except Exception as e:
        print(f"Error removing file: {e}")
else:
    print("File does not exist via \\\\.\\ prefix")
    
if os.path.exists(normal_path):
     print("File exists (accessed via normal path)")
     try:
        os.remove(normal_path)
        print("Successfully removed file.")
     except Exception as e:
        print(f"Error removing file: {e}")
