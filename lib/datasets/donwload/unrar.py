import os
import patoolib

def extract_rar_files(directory):
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a .rar file
            if file.endswith(".rar"):
                rar_path = os.path.join(root, file)
                # Define the output directory (same as the .rar file name without the extension)
                output_dir = rar_path.rsplit('.', 1)[0]
                # Create the output directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # Extract the .rar file
                try:
                    patoolib.extract_archive(rar_path, outdir=output_dir)
                    print(f"Extracted: {rar_path} to {output_dir}")
                except Exception as e:
                    print(f"Error extracting {rar_path}: {e}")

if __name__ == "__main__":
    # Specify the directory to search for .rar files
    directory_to_search = "../data"
    extract_rar_files(directory_to_search)
