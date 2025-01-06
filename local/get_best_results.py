import re
from pathlib import Path
from collections import defaultdict
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="/mgData1/yangb/icefall/egs/omini/ASR/zipformer/multilingual_8asr_16000h_280M_online_langtag/streaming/greedy_search",
    )
    
    parser.add_argument(
        "--num-languages",
        type=int,
        default=8,
    )

    return parser


def extract(results_dir, num_languages):
    results = defaultdict(dict)
    for file in results_dir:
        name = file.name
        info = re.match(r'wer-summary-(.*)-greedy_search-(.*)-chunk-.*', name)
        testset, ckpt = info[1], info[2]
        with open(file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 1:
                    wer = float(line.strip().split('\t')[1])
        results[ckpt][testset] = wer
    
    avg_results = defaultdict(float)
    for ckpt in results:
        if len(results[ckpt]) != num_languages:
            continue
        for testset in results[ckpt]:
            avg_results[ckpt] += results[ckpt][testset]
        avg_results[ckpt] = avg_results[ckpt] / num_languages

    best_result = sorted(avg_results.items(), key=lambda item: item[1])
    for res in best_result:
        print(f'ckpt: {res[0]} avg: {res[1]}')
        for testset in results[res[0]]:
            print(f'{testset} {results[res[0]][testset]}')
        print('------------------------------------------')


def get_best_results(res_dir, num_languages):
    wer_files = []
    for file in res_dir.iterdir():
        if file.name.startswith('wer'):
            wer_files.append(file)
    extract(wer_files, num_languages)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    get_best_results(Path(args.results_dir), args.num_languages)