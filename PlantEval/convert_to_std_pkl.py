"""
Convert dill-based pickle files to standard pickle format.
"""
import pickle as pkl
import dill
from pathlib import Path

def convert_to_std_pkl(input_path, output_path):
    """Convert a dill pickle to standard pickle."""
    print(f"Loading from: {input_path}")
    
    # Load with dill
    with open(input_path, 'rb') as f:
        benchmark = dill.load(f)
    
    # Extract realizations and convert to simple dicts
    realizations = []
    for i, real in enumerate(benchmark.realizations):
        print(f"Processing realization {i+1}/{len(benchmark.realizations)}...")
        real_dict = {
            'X_train': real.X_train,
            'X_test': real.X_test,
            't_train': real.t_train,
            't_test': real.t_test,
            'y_train': real.y_train,
            'true_cid': real.true_cid,
        }
        realizations.append(real_dict)
    
    # Save with standard pickle
    data = {'realizations': realizations}
    
    print(f"Saving to: {output_path}")
    with open(output_path, 'wb') as f:
        pkl.dump(data, f)
    
    print(f"Conversion complete! Created {len(realizations)} realizations")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python convert_to_std_pkl.py <dataset_name>")
        print("Example: python convert_to_std_pkl.py CID_10_ints_10_reals_four_std_uniform")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    base_dir = Path(__file__).parent / "PlantsEval" / "plant_data"
    
    input_path = base_dir / f"{dataset_name}.pkl"
    output_path = base_dir / f"{dataset_name}_std.pkl"
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if output_path.exists():
        print(f"Warning: Output file already exists: {output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    convert_to_std_pkl(input_path, output_path)
