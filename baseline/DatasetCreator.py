import os
import pandas as pd
from multiprocessing import Pool

batch_dir = "baseline/designite_output"
output_dir = "baseline/reduced_csv"

os.makedirs(output_dir, exist_ok=True)



def reduce_csv(file_path):
    df = pd.read_csv(file_path)

    if df.empty:
        return None 
    project_name = df['Project Name'].iloc[0]  
    label = project_name.rsplit('_', 1)[0] 
    unique_id = os.path.basename(os.path.dirname(file_path))

    loc_sum = df['LOC'].sum()
    cc_max = df['CC'].max()
    pc_max = df['PC'].max()

    return [unique_id, label, loc_sum, cc_max, pc_max]

def process_batch(batch_folder):
    batch_path = os.path.join(batch_dir, batch_folder)
    reduced_rows = []

    for snippet_folder in os.listdir(batch_path):
        snippet_path = os.path.join(batch_path, snippet_folder, 'MethodMetrics.csv')
        if os.path.isfile(snippet_path):
            reduced_row = reduce_csv(snippet_path)
            if reduced_row is not None:
                reduced_rows.append(reduced_row)

    output_path = os.path.join(output_dir, f"{batch_folder}_reduced.csv")
    reduced_df = pd.DataFrame(reduced_rows, columns=["UniqueID", "Label", "LOC", "CC", "PC"])
    reduced_df.to_csv(output_path, index=False)

def main():
    batch_folders = [folder for folder in os.listdir(batch_dir) if os.path.isdir(os.path.join(batch_dir, folder))]

    process_batch(batch_folders[0])

    with Pool(processes=6) as pool:
        pool.map(process_batch, batch_folders)

    final_df = pd.concat([pd.read_csv(os.path.join(output_dir, file)) for file in os.listdir(output_dir)], ignore_index=True)
    final_df.to_csv("baseline/final_dataset.csv", index=False)


if __name__ == '__main__':
    main()