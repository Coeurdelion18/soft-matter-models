from pathlib import Path
from datasets.lammps_parser import lammps_to_numpy
from tqdm import tqdm

input_folder = Path("data/raw/config")
output_folder = Path("data/raw/config_npy")

output_folder.mkdir(parents=True, exist_ok=True)

for item in tqdm(list(input_folder.iterdir())):
    if not item.is_file():
        continue

    output_path = "data/raw/config_npy/" + item.stem + ".npy"
    
    img = lammps_to_numpy(
        input_file=item, 
        output_file=output_path,
        grid_size=192,
        sigma_scale=1.0,   # sigma = diameter/2
        normalize=False
    )