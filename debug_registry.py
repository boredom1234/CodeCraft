import pickle
from pathlib import Path

print("Debugging registry file...")
registry_file = Path(".codeai") / "registry.pkl"

if not registry_file.exists():
    print("Registry file does not exist!")
else:
    print(f"Registry file exists at {registry_file}")
    
    with open(registry_file, 'rb') as f:
        projects = pickle.load(f)
    
    print(f"Full registry content: {projects}")
    
    active = projects.get('active', None)
    print(f"Active project: {active}")
    
    print("\nAll projects:")
    project_count = 0
    for name, info in projects.items():
        if name != 'active':
            project_count += 1
            print(f"  - Project name: '{name}'")
            print(f"    Is active: {name == active}")
            print(f"    Created: {info.get('created', 'unknown')}")
            print(f"    Raw info: {info}")
    
    print(f"\nTotal projects found: {project_count}") 