import os
import subprocess
from multiprocessing import Pool

batch_dir = "baseline/java_batches"
designite_output_dir = "baseline/designite_output"

os.makedirs(designite_output_dir, exist_ok=True)

def run_designite_for_batch(batch_folder):
    batch_path = os.path.join(batch_dir, batch_folder)

    batch_output_path = os.path.join(designite_output_dir, batch_folder)
    os.makedirs(batch_output_path, exist_ok=True)

    for snippet_folder in os.listdir(batch_path):
        snippet_path = os.path.join(batch_path, snippet_folder)

        snippet_output_path = os.path.join(batch_output_path, snippet_folder)
        os.makedirs(snippet_output_path, exist_ok=True)

        designite_command = [
            "java", "-jar", "DesigniteJava/target/DesigniteJava.jar",
            "-i", snippet_path,
            "-o", snippet_output_path
        ]

        print(f"Running Designite on {snippet_path}...")
        subprocess.run(designite_command, check=True)

if __name__ == '__main__':
    batch_folders = [folder for folder in os.listdir(batch_dir) if os.path.isdir(os.path.join(batch_dir, folder))]

    num_workers = 6  
    with Pool(num_workers) as pool:
        pool.map(run_designite_for_batch, batch_folders)

    print("Designite processing completed for all batches.")
