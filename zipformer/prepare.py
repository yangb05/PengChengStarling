import re
import argparse
from pathlib import Path
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

import json
import yaml
import torch
import unicodedata
import pandas as pd
from tqdm import tqdm
from lhotse import (
    Recording, 
    SupervisionSegment, 
    RecordingSet, 
    SupervisionSet, 
    fix_manifests, 
    validate_recordings_and_supervisions, 
    CutSet,
    KaldifeatFbank, 
    KaldifeatFbankConfig
)
from lhotse.recipes.utils import read_manifests_if_cached
from icefall.utils import AttributeDict
import torchaudio


# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config-file",
        type=str,
        default="config_data/ru-cv_15h.yaml",
        help="The config file.",
    )

    return parser


class Prepare():
    """
    This class is used to compute the fbank feature from the audio. The produced data.list and fbank is in corpus_dir.
    """
    def __init__(
        self,
        params,
        
    ):
        self.corpus_dir = Path(params.corpus_dir)
        self.raw_data = Path(params.raw_data)
        self.datalist_path = Path(params.datalist_path)
        self.manifest_dir = Path(params.manifest_dir)
        self.fbank_dir = Path(params.fbank_dir)
        self.prefix = params.prefix
        self.partition = params.partition
        self.suffix = params.suffix
        self.format = params.format
        self.num_workers = params.num_workers
        self.batch_duration = params.batch_duration
        self.speed_perturb = params.speed_perturb
        self.language = params.language
        self.langtag = params.langtag      
            
    def normalize_text(self, text):
        # apply NFKC
        text = unicodedata.normalize('NFKC', text)
        # remove unprintable chars
        text = text.replace(u'\ufe0f', '')
        text = text.replace(u'\u200b', '')
        text = text.replace(u'\u200c', '')
        text = text.replace(u'\u202d', '')
        # remove silence symbols
        ar_punctuations = '''`؛<>_()*&^][ـ،/:"؟.,'{}~¦|!”…“–ـ'''
        en_zh_punctuations = '''!"#$&'()*,./:;?[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·'''
        other_silence_symbols = '⠀–‐‒—⁃₽∙√⋅█■◊●☀️♂️♡♥♫⚽️✆✝️❖⦁一﻿￼'
        all_silence_symbols = ar_punctuations + en_zh_punctuations + other_silence_symbols
        text = text.translate(str.maketrans('', '', all_silence_symbols))
        # whitespaces
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        # uppercase
        text = text.upper()
        # add langtag
        text = self.langtag + text
        return text
    
    def normalize_audio(self, audio, seg_dir):
        audio_list = []
        audio_path = self.corpus_dir / 'audio' / audio['path']
        if audio_path.is_file():
            waveform, sampling_rate = torchaudio.load(audio_path)
            if waveform.size(0) > 1: # extract first channel
                waveform = waveform[0].view(1, -1)
            if sampling_rate != 16000: # resample to 16000
                waveform = torchaudio.functional.resample(waveform, sampling_rate, 16000)
            # extract segments
            for raw_segment_info in tqdm(audio['segments'], desc='making segments'): # split into segments
                segment_info = self.extract_segment(seg_dir, waveform, raw_segment_info)
                audio_list.append(segment_info)
        return audio_list

    def extract_segment(self, seg_dir, waveform, raw_segment_info):
        seg_file = raw_segment_info['sid'] + '.wav'
        seg_path = seg_dir / seg_file
        segment_info = {
            'key': raw_segment_info['sid'],
            'wav': str(seg_path),
            'txt': Prepare.normalize_text(raw_segment_info['text_tn']),
            'language': self.language
        }
        if not seg_path.is_file(): 
            begin_time = float(raw_segment_info['begin_time'])
            end_time = float(raw_segment_info['end_time'])
            frame_offset = int(begin_time * 16000)
            num_frames = int((end_time - begin_time) * 16000)
            seg_wav = waveform[:, frame_offset : min(frame_offset + num_frames, waveform.size(1))]
            torchaudio.save(seg_path, seg_wav, 16000, encoding="PCM_S", bits_per_sample=16)
        return segment_info
        
    def make_datatang_cp_list(self):
        '''
        data.list should at least include key, language, wav and txt items.
        '''
        if self.datalist_path.is_file():
            print('data.list exist, skip making data.list!')
        else:
            data_list = []
            seen = set()
            # read data info
            for category in self.corpus_dir.iterdir():
                for speaker in tqdm(category.iterdir(), desc=f'Reading segments'):
                    for segment in speaker.iterdir():
                        segment_info = {}
                        segment_id = segment.stem
                        if segment_id not in seen:
                            seen.add(segment_id)
                            segment_info['key'] = segment_id
                            segment_info['language'] = self.language
                            wav_file = segment.with_suffix('.wav')
                            txt_file = segment.with_suffix('.txt')
                            meta_file = segment.with_suffix('.metadata')
                            segment_info['wav'] = str(wav_file)
                            # read and normalize text
                            with open(txt_file, 'r', encoding='utf8') as f1:
                                text = f1.readline().strip()
                            text = Prepare.normalize_text(text)
                            segment_info['txt'] = text
                            # read meta info
                            with open(meta_file, 'r', encoding='utf8') as f2:
                                for line in f2.readlines():
                                    meta = line.strip().split('\t')
                                    if len(meta) < 2:
                                        continue
                                    segment_info.update({meta[0]: meta[1]})
                            # build data.list
                            data_list.append(segment_info)
            # save to data.list
            with open(self.datalist_path, 'w', encoding='utf8') as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False)+'\n')
    
    def make_datatang_test_list(self):
        if self.datalist_path.is_file():
            print('data.list exist, skip making data.list!')
        else:
            data_list = []
            with open(self.raw_data, 'r', encoding='utf8') as f:
                rawdata = json.load(f)
            seg_dir = self.corpus_dir / 'formatted'
            seg_dir.mkdir(exist_ok=True)
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                for audio in tqdm(rawdata['audios'], desc='normalizing audios'):
                    data_list += self.normalize_audio(audio, seg_dir)
            # save to data.list
            self.datalist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.datalist_path, 'w', encoding='utf8') as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False)+'\n')    
    
    def make_cv_list(self):
        if self.datalist_path.is_file():
            print('data.list exist, skip making data.list!')
        else:
            data_list = []
            rawdata = pd.read_csv(self.corpus_dir / 'test.tsv', sep='\t', index_col=False)
            for idx, row in tqdm(rawdata.iterrows(), desc='reading segments'):
                segment_info = {}    
                segment_info['language'] = self.language
                segment_info['txt'] = Prepare.normalize_text(row.pop('sentence'))
                wav_file = row.pop('path').replace('.mp3', '.wav')
                wav_path = self.corpus_dir / 'ru_test_0' / wav_file
                segment_info['wav'] = str(wav_path)
                segment_info['key'] = wav_file.split('.')[0].rsplit('_', 1)[1]
                [segment_info.update({k:v}) for k,v in row.items() if not pd.isna(v) ]
                data_list.append(segment_info)
            # save to data.list
            with open(self.datalist_path, 'w', encoding='utf8') as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False)+'\n')
            
    def make_rostt_list(self):
        '''
        data.list should at least include key, language, wav and txt items.
        '''
        if self.datalist_path.is_file():
            print('data.list exist, skip making data.list!')
        else:
            data_list = []
            seen = set()
            # read data info
            for category in self.corpus_dir.iterdir():
                for speaker in tqdm(category.iterdir(), desc=f'Reading segments'):
                    for segment in speaker.iterdir():
                        segment_info = {}
                        segment_id = segment.stem
                        if segment_id not in seen:
                            seen.add(segment_id)
                            segment_info['key'] = segment_id
                            segment_info['language'] = self.language
                            wav_file = segment.with_suffix(self.format)
                            txt_file = segment.with_suffix('.txt')
                            if not txt_file.is_file():
                                continue
                            segment_info['wav'] = str(wav_file)
                            # read and normalize text
                            with open(txt_file, 'r', encoding='utf8') as f1:
                                text = f1.readline().strip()
                            text = Prepare.normalize_text(text)
                            segment_info['txt'] = text
                            # build data.list
                            data_list.append(segment_info)
            # save to data.list
            self.datalist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.datalist_path, 'w', encoding='utf8') as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False)+'\n')
    
    def make_msr86k_list(self):
        if self.datalist_path.is_file():
            print('data.list exist, skip making data.list!')
        else:
            data_list = []
            train_aids = []
            for audio_file in (self.corpus_dir).iterdir():
                if audio_file.suffix == self.format:
                    train_aids.append(audio_file.stem)
            with open(self.raw_data, 'r', encoding='utf8') as raw_file:
                raw_data = json.load(raw_file)
            seg_dir = self.corpus_dir / 'formatted'
            seg_dir.mkdir(exist_ok=True) 
            
            def normalize_audio(audio):
                audio_list = []
                if audio['aid'] in train_aids:
                    audio_path = self.corpus_dir / (audio['aid'] + self.format)
                    waveform, sampling_rate = torchaudio.load(audio_path)
                    if waveform.size(0) > 1: # extract first channel
                        waveform = waveform[0].view(1, -1)
                    if sampling_rate != 16000: # resample to 16000
                        waveform = torchaudio.functional.resample(waveform, sampling_rate, 16000)
                    with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                        for segment_info in tqdm(ex.map(
                            self.extract_segment, 
                            repeat(seg_dir), 
                            repeat(waveform), 
                            audio['segments']),
                            desc='making segments'): # split into segments
                            audio_list.append(segment_info)
                return audio_list
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                for audio in tqdm(raw_data['audios'], desc='normalizing audios'):
                    data_list += normalize_audio(audio)
            # save to data.list
            self.datalist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.datalist_path, 'w', encoding='utf8') as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False)+'\n')
    
    def make_librispeech_list(self):
        if self.datalist_path.is_file():
            print('data.list exists, skip making data.list!')
        else:
            data_list = []
            # read jsonl
            with open(self.raw_data, 'r', encoding='utf8') as f:
                for line in f.readlines():
                    segment_info = {}  
                    data = json.loads(line)
                    seg_path = self.corpus_dir / data.pop('audio_filepath')
                    segment_info['key'] = seg_path.stem
                    segment_info['wav'] = str(seg_path)
                    segment_info['txt'] = Prepare.normalize_text(data.pop('text'))
                    segment_info.update({k:v for k,v in data.items()})
                    segment_info['language'] = self.language
                    data_list.append(segment_info)
            # save to data.list
            with open(self.datalist_path, 'w', encoding='utf8') as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False)+'\n')
    
    def make_wenetspeech_list(self):
        if self.datalist_path.is_file():
            print('data.list exists, skip making data.list!')
        else:
            data_list = []
            # read jsonl
            cuts = CutSet.from_file(self.raw_data)
            for cut in cuts:
                segment_info = {}
                segment_info['key'] = cut.recording.id
                segment_info['wav'] = str(self.corpus_dir / f'{cut.recording.id}.wav')
                segment_info['txt'] = Prepare.normalize_text(cut.supervisions[0].text)
                segment_info['language'] = self.language
                data_list.append(segment_info)
            # save to data.list
            with open(self.datalist_path, 'w', encoding='utf8') as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False)+'\n')   
    
    def make_gigaspeech2_list(self):
        if self.datalist_path.is_file():
            print('data.list exists, skip making data.list!')
        else:
            self.datalist_path.parent.mkdir(parents=True, exist_ok=True)
            data_list = []
            # read tsv
            empty_lines = []
            with open(self.raw_data, 'r', encoding='utf8') as f:
                for line in f.readlines():
                    segment_info = {}
                    try:
                        key, text = line.strip().split('\t')
                    except:
                        empty_lines.append(line)
                        continue
                    dirs = key.split('-')
                    segment_info['key'] = key
                    if len(dirs) == 3:
                        segment_info['wav'] = str(self.corpus_dir / dirs[0] / dirs[1] / f'{key}.wav')
                    else:
                        segment_info['wav'] = str(self.corpus_dir / dirs[0] / f'{key}.wav')
                    if not Path(segment_info['wav']).is_file():
                        continue
                    segment_info['txt'] = Prepare.normalize_text(text)
                    segment_info['language'] = self.language
                    data_list.append(segment_info)
            print(empty_lines)
            # save to data.list
            with open(self.datalist_path, 'w', encoding='utf8') as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False)+'\n')
    
    @staticmethod
    def make_recording(segment_info):
        recording = Recording.from_file(segment_info.pop('wav'), recording_id=segment_info.pop('key'), force_opus_sampling_rate=16000)
        return recording

    @staticmethod
    def make_supervision(recording, segment_info):
        supervision = SupervisionSegment(
            id=recording.id,
            recording_id=recording.id,
            start=0,
            duration=recording.duration,
            channel=0,
            text=segment_info.pop('txt'),
            language=segment_info.pop('language'),
            custom=segment_info
        )
        return supervision

    @staticmethod
    def make_manifest(line):
        segment_info = json.loads(line.strip())
        recording = Prepare.make_recording(segment_info)
        supervision = Prepare.make_supervision(recording, segment_info)
        return recording, supervision

    def make_manifests(self):
        """
        Make manifests from recordings and supervisions.
        """
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        recording_path = self.manifest_dir / f'{self.prefix}_recordings_{self.partition}.{self.suffix}'
        supervision_path = self.manifest_dir / f'{self.prefix}_supervisions_{self.partition}.{self.suffix}'
        if recording_path.is_file() and supervision_path.is_file():
            print(f'manifests exist, skip making manifests!')
        else:
            recordings = []
            supervisions = []
            assert (self.datalist_path).is_file(), 'make_*_list function should be performed at first!'
            # make manifests
            with open(self.datalist_path, 'r', encoding='utf8') as f:
                with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                    for recording, supervision in tqdm(
                        ex.map(Prepare.make_manifest, f.readlines()), 
                        desc=f'making manifests'
                    ):
                        if recording.duration > 0.0:
                            recordings.append(recording)
                            supervisions.append(supervision)
            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)
            recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
            validate_recordings_and_supervisions(recording_set, supervision_set)
            recording_set.to_file(recording_path)
            supervision_set.to_file(supervision_path)

    def compute_fbank(self):
        cut_path = self.fbank_dir / f'{self.prefix}_cuts_{self.partition}.{self.suffix}'
        feat_path = self.fbank_dir / f'{self.prefix}_feats_{self.partition}.lca'
        if cut_path.is_file() and feat_path.is_file():
            print(f'fbank exist, skip computing fbank!')
        else:
            # read manifests
            manifests = read_manifests_if_cached(
                dataset_parts=self.partition,
                output_dir=self.manifest_dir,
                prefix=self.prefix,
                suffix=self.suffix,
            )
            # make extractor
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda", 0)
            extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))
            # make cuts
            cut_set = CutSet.from_manifests(
                recordings=manifests[self.partition]['recordings'],
                supervisions=manifests[self.partition]['supervisions'],
            )
            # make features
            self.fbank_dir.mkdir(parents=True, exist_ok=True)
            cut_set = cut_set.compute_and_store_features_batch(
                extractor=extractor,
                storage_path=f"{self.fbank_dir}/{self.prefix}_feats_{self.partition}",
                num_workers=self.num_workers,
                batch_duration=self.batch_duration,
                overwrite=True,
            )
            print(f'Saving to {cut_path}')
            cut_set.to_file(cut_path)


def run_prepare(params):
    prepare = Prepare(params)
    if 'datatang_cp' in params.config_file:
        prepare.make_datatang_cp_list()
    elif 'datatang_test' in params.config_file:
        prepare.make_datatang_test_list()
    elif 'commonvoice' in params.config_file:
        prepare.make_cv_list()
    elif 'msr86k' in params.config_file:
        prepare.make_msr86k_list()
    elif 'librispeech' in params.config_file or 'golos' in params.config_file:
        prepare.make_librispeech_list()
    elif 'wenetspeech' in params.config_file:
        prepare.make_wenetspeech_list()
    elif 'gigaspeech2' in params.config_file:
        prepare.make_gigaspeech2_list()
    else:
        prepare.make_rostt_list()
    prepare.make_manifests()
    prepare.compute_fbank()


def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config_file, encoding='utf8') as f:
        params = AttributeDict(yaml.safe_load(f))
    params.corpus_dir = Path(params.corpus_dir)
    params.update(vars(args))
    run_prepare(params)
    

if __name__ == '__main__':
    main()