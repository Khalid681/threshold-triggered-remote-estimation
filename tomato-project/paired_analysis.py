from pathlib import Path
import argparse
import pandas as pd
from aoi_project.analysis import paired_comparison


def main():
    parser = argparse.ArgumentParser(description='Node-clustered paired inference; removes consistent legacy deadline copies')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--input', type=Path, help='Optional legacy or normalized accuracy CSV')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.run_dir is None and args.input is None:
        parser.error('Specify --run-dir or --input')
    source = args.input or args.run_dir/'raw'/'estimation_results.csv'
    output = args.output or (args.run_dir/'summaries'/'paired_comparison.csv' if args.run_dir else None)
    if output is None:
        parser.error('--input requires --output to protect previous results')
    paired, nodes = paired_comparison(pd.read_csv(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output, index=False)
    nodes.to_csv(output.with_name('node_level_effects.csv'), index=False)
    print(f'{len(paired)} comparisons saved to {output}; all intervals use independent nodes')


if __name__ == '__main__':
    main()
