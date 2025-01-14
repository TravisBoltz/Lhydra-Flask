from pathlib import Path
from typing import Dict

def setup_paths() -> Dict[str, Path]:
    """Set up all project paths."""
    # Get root directory (Lhydra_rs)
    root_dir = Path(__file__).resolve().parent
    
    paths = {
        'root': root_dir,
        'data': root_dir / 'data',
        'processed_data': root_dir / 'data' / 'processed_data',
        'models': root_dir / 'checkpoints',
        'encoder': root_dir / 'data' / 'processed_data' / 'encoder.pt',
        'test_data': root_dir / 'data' / 'processed_data' / 'test_data.csv',
        'train_data': root_dir / 'data' / 'processed_data' / 'train_data.csv'
    }
    
    # Create required directories
    for path in ['processed_data', 'models']:
        paths[path].parent.mkdir(parents=True, exist_ok=True)
    return paths
