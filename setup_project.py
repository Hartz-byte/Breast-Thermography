import os

project_structure = {
    "data": ["raw/README.md", "processed/README.md"],
    "notebooks": [],
    "src": ["__init__.py", "data_prep.py", "model.py", "train.py", "evaluate.py", "visualize.py"],
    "reports": ["README.md"],
    "outputs": ["README.md"],
    "configs": ["config.yaml"]
}

def create_structure(base_path="."):
    for folder, files in project_structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)

        for file in files:
            file_path = os.path.join(folder_path, file)
            # Create nested directories if needed
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Create the file
            with open(file_path, "w") as f:
                f.write(f"# {file}")

    print(f"Project structure created under: {os.path.abspath(base_path)}")

if __name__ == "__main__":
    create_structure()
