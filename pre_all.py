import os

def preprocess(dir):
    files = os.listdir(dir)
    flag = False
    for f in files:
        f_path = os.path.join(dir, f)
        if os.path.isdir(f_path):
            preprocess(f_path)
            flag = True
    if not flag:
        if "preprocessed_data.json" in files:
            print(f"Skipping {dir}, already preprocessed.")
        else:
            py = "/home/senyang/miniconda3/envs/rl/bin/python"
            model = "SSP" if "SSP" in dir else "MSP"
            cmd = f"{py} /home/senyang/geographical-decentralization-simulation/preprocess_data.py -d /home/senyang/geographical-decentralization-simulation/data -o {dir} -m {model}"
            print(f"Preprocessing {dir} with model {model}...")
            os.system(cmd)

def copy_file(src, prefix, dst):
    files = os.listdir(src)
    flag = False
    for f in files:
        f_path = os.path.join(src, f)
        if os.path.isdir(f_path):
            copy_file(f_path, prefix, dst)
            flag = True
    
    if not flag:
        if "preprocessed_data.json" in files:
            dst_dir = src.replace(prefix, dst).replace("validators_1000_slots_10000_", "")
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            src_path = os.path.join(src, "preprocessed_data.json")
            dst_path = os.path.join(dst_dir, "data.json")
            os.system(f"cp {src_path} {dst_path}")
            print(f"Copied preprocessed_data.json from {src} to {dst_path}")
    

if __name__ == "__main__":
    base_dir = "/home/senyang/geographical-decentralization-simulation/output"
    # preprocess(base_dir)
    copy_file(base_dir, base_dir, "/home/senyang/geographical-decentralization-simulation/dashboard/simulations")
