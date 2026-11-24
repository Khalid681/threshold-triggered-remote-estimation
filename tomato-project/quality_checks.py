import argparse
from aoi_project.quality import check_run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    report = check_run(args.run_dir)
    print('Quality: '+('PASS' if report['passed'] else 'FAIL'))
    for name, passed in report['checks'].items():
        if not passed:
            print('FAILED: '+name)
    raise SystemExit(0 if report['passed'] else 1)
