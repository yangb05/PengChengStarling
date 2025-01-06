#  Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
import logging
from functools import lru_cache
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    AudioSamples,
    OnTheFlyFeatures,
)
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class AsrDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, params):
        self.args = params

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        transforms = []
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            logging.info("About to get Musan cuts")
            cuts_musan = load_manifest(self.args.manifest_dir / "musan_cuts.jsonl.gz")
            transforms.append(
                CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        if self.args.concatenate_cuts:
            logging.info(
                f"Using cut concatenation with duration factor "
                f"{self.args.duration_factor} and gap {self.args.gap}."
            )
            # Cut concatenation should be the first transform in the list,
            # so that if we e.g. mix noise in, it will fill the gaps between
            # different utterances.
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")
            # Set the value of num_frame_masks according to Lhotse's version.
            # In different Lhotse's versions, the default of num_frame_masks is
            # different.
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            logging.info(f"Num frame mask: {num_frame_masks}")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")
        train = K2SpeechRecognitionDataset(
            input_strategy=eval(self.args.input_strategy)(),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
        )

        if self.args.on_the_fly_feats:
            # NOTE: the PerturbSpeed transform should be added only if we
            # remove it from data prep stage.
            # Add on-the-fly speed perturbation; since originally it would
            # have increased epoch size by 3, we will apply prob 2/3 and use
            # 3x more epochs.
            # Speed perturbation probably should come first before
            # concatenation, but in principle the transforms order doesn't have
            # to be strict (e.g. could be randomized)
            # transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2/3)] + transforms   # noqa
            # Drop feats to be on the safe side.
            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
            )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 2000,
                shuffle_buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        logging.info("About to create valid dataset")
        if self.args.on_the_fly_feats:
            valid = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                return_cuts=self.args.return_cuts,
            )
        else:
            valid = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                return_cuts=self.args.return_cuts,
            )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.info("About to create valid dataloader")
        valid_dl = DataLoader(
            valid,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def train_cuts(self) -> CutSet:
        if len(self.args.train_cuts) == 1:
            cut_name = self.args.train_cuts[0].split(".")[0]
            logging.info(f"About to get {cut_name}")
            return load_manifest_lazy(
                self.args.manifest_dir / self.args.train_cuts[0]
            )
        else: # handle multiple datasets
            traincuts = []
            for cut_file in self.args.train_cuts:
                cut_name = cut_file.split(".")[0]
                logging.info(f"About to get {cut_name}")
                traincuts.append(load_manifest_lazy(self.args.manifest_dir / cut_file))
            return CutSet.mux(*traincuts, weights=[len(cut) for cut in traincuts])
    
    @lru_cache()
    def finetune_cuts(self) -> CutSet:
        if len(self.args.finetune_cuts) == 1:
            cut_name = self.args.finetune_cuts[0].split(".")[0]
            logging.info(f"About to get {cut_name}")
            return load_manifest_lazy(
                self.args.manifest_dir / self.args.finetune_cuts[0]
            )
        else: # handle multiple datasets
            ftcuts = []
            for cut_file in self.args.finetune_cuts:
                cut_name = cut_file.split(".")[0]
                logging.info(f"About to get {cut_name}")
                ftcuts.append(load_manifest_lazy(self.args.manifest_dir / cut_file))
            return CutSet.mux(*ftcuts, weights=[len(cut) for cut in ftcuts])
    
    @lru_cache()
    def valid_cuts(self) -> CutSet:
        cut_name = self.args.valid_cuts.split(".")[0]
        logging.info(f"About to get {cut_name}")
        return load_manifest_lazy(
            self.args.manifest_dir / self.args.valid_cuts
        )
    
    @lru_cache()
    def test_cuts(self) -> CutSet:
        testcuts = {}
        for cut_file in self.args.test_cuts:
            cut_name = cut_file.split(".")[0]
            logging.info(f"About to get {cut_name}")
            testcuts.update({
                cut_name: load_manifest_lazy(self.args.manifest_dir / cut_file)
            })
        return testcuts