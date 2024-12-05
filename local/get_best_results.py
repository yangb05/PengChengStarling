import re
from pathlib import Path
from collections import defaultdict


# a = 'wer-summary-gigaspeech_cuts_test-greedy_search-epoch-20-avg-10-chunk-16-left-context-128-use-averaged-model.txt'
# groups = re.match(r'wer-summary-(.*)-greedy_search-(.*)-chunk-.*', a)
# print(groups[1])
# print(groups[2])


def extract(wer_files):
    results = defaultdict(dict)
    for file in wer_files:
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
        assert len(results[ckpt]) == 8
        for testset in results[ckpt]:
            avg_results[ckpt] += results[ckpt][testset]
        avg_results[ckpt] = avg_results[ckpt] / 8

    best_result = sorted(avg_results.items(), key=lambda item: item[1])[0]
    print(f'best ckpt: {best_result[0]} avg: {best_result[1]}')
    
    for testset in results[best_result[0]]:
        print(f'{testset} {results[best_result[0]][testset]}')


def get_best_results(res_dir):
    wer_files = []
    for file in res_dir.iterdir():
        if file.name.startswith('wer'):
            wer_files.append(file)
    extract(wer_files)


if __name__ == '__main__':
    get_best_results(Path('/mgData1/yangb/icefall/egs/omini/ASR/zipformer/multilingual_8asr_16000h_280M_online_langtag/streaming/greedy_search'))