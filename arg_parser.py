import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--repo_id", type=str)
    parser.add_argument("--prompt", type=str)
    
    return parser.parse_args()