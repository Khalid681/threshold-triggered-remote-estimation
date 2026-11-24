import argparse
from aoi_project.plotting import plot_run

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Plot all nodes, separate thresholds, relative deadlines and node-cluster intervals')
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    manifest=plot_run(args.run_dir)
    print(f'Saved {len(manifest["files"])} files; nodes: {manifest["node_ids"]}')
