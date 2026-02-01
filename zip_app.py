import shutil
import os

def zip_app():
    source_dir = os.path.join('dist', 'MeetingTranslator')
    output_filename = 'MeetingTranslator'
    
    print(f"Zipping {source_dir} to {output_filename}.zip...")
    
    shutil.make_archive(output_filename, 'zip', root_dir='dist', base_dir='MeetingTranslator')
    
    print("Zip created successfully.")

if __name__ == "__main__":
    zip_app()
