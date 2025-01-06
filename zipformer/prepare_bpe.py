import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import shutil
import yaml
import sentencepiece as spm
from tqdm import tqdm
from lhotse import CutSet
from icefall.utils import tokenize_by_CJK_char
from icefall.lexicon import write_lexicon
import k2
import pythainlp
import nagisa
from icefall.utils import AttributeDict


Lexicon = List[Tuple[str, List[str]]]

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


class PrepareBPE():
    """
    This class is used to prepare the bpe model and tokens.txt
    """
    def __init__(self, params):
        self.bpe_cuts_dir = Path(params.bpe_cuts_dir)
        self.bpe_cuts = params.bpe_cuts
        self.vocab_size = params.vocab_size
        self.lang_dir = Path(params.lang_dir)
        self.langtags = params.langtags
        self.model_type = params.model_type
        self.character_coverage = params.character_coverage
        self.input_sentence_size = params.input_sentence_size
    
    @staticmethod
    def make_words(transcripts):
        special_words = ['!SIL', '<SPOKEN_NOISE>', '<UNK>']
        # combine
        words = sorted(set(' '.join(transcripts).split() + special_words))
        # add id
        word2id = ['<eps> 0']
        for idx, word in enumerate(words, start=1):
            if word == '<s>' or word == '</s>':
                raise ValueError(f'{word} is in the vocabulary!')
            word2id.append(f'{word} {idx}')
        word2id.append(f'#0 {idx+1}')
        word2id.append(f'<s> {idx+2}')
        word2id.append(f'</s> {idx+3}')
        return word2id 
    
    def train_bpe_model(self):
        if not (self.lang_dir / 'transcript_words.txt').is_file():
            if self.bpe_cuts:
                cuts = CutSet()
                for bpe_cut in self.bpe_cuts:
                    cut_path = self.bpe_cuts_dir / bpe_cut
                    cuts += CutSet.from_file(cut_path)
            else:        
                cut_path = self.fbank_dir / f'{self.prefix}_cuts_{self.partition}.{self.suffix}'
                cuts = CutSet.from_file(cut_path)
            self.lang_dir.mkdir(parents=True, exist_ok=True)
            # write transcripts
            transcripts = []
            for cut in tqdm(cuts, desc='word segmentation'):
                if cut.supervisions[0].language == 'Chinese':
                    transcripts.append(tokenize_by_CJK_char(cut.supervisions[0].text))
                elif cut.supervisions[0].language == 'Thai':                    
                    result = pythainlp.word_tokenize(cut.supervisions[0].text, keep_whitespace=False)
                    transcripts.append(' '.join(result))
                elif cut.supervisions[0].language == 'Japanese':
                    result = nagisa.tagging(cut.supervisions[0].text)
                    transcripts.append(' '.join(result.words))
                else:
                    transcripts.append(cut.supervisions[0].text)
            with open(self.lang_dir / 'transcript_words.txt', 'w', encoding='utf8') as tf:
                for transcript in transcripts:
                    tf.write(transcript+'\n')
        # write words
        if not (self.lang_dir / 'words.txt').is_file():
            word2id = PrepareBPE.make_words(transcripts)
            with open(self.lang_dir / 'words.txt', 'w', encoding='utf8') as wf:
                for word in word2id:
                    wf.write(word+'\n')
        # train bpe
        model_prefix = f"{self.lang_dir}/{self.model_type}_{self.vocab_size}"
        user_defined_symbols = ["<blk>", "<sos/eos>"]
        if self.langtags:
            user_defined_symbols += self.langtags
        unk_id = len(user_defined_symbols)
        model_file = Path(model_prefix + ".model")
        if not model_file.is_file():
            spm.SentencePieceTrainer.train(
                input=self.lang_dir / 'transcript_words.txt',
                vocab_size=self.vocab_size,
                model_type=self.model_type,
                model_prefix=model_prefix,
                input_sentence_size=self.input_sentence_size,
                character_coverage=self.character_coverage,
                user_defined_symbols=user_defined_symbols,
                unk_id=unk_id,
                bos_id=-1,
                eos_id=-1,
            )
        else:
            print(f"{model_file} exists - skipping")
            return

        shutil.copyfile(model_file, f"{self.lang_dir}/bpe.model")
    
    @staticmethod
    def generate_lexicon(
        model_file: str, words: List[str], oov: str
    ) -> Tuple[Lexicon, Dict[str, int]]:
        """Generate a lexicon from a BPE model.

        Args:
        model_file:
            Path to a sentencepiece model.
        words:
            A list of strings representing words.
        oov:
            The out of vocabulary word in lexicon.
        Returns:
        Return a tuple with two elements:
            - A dict whose keys are words and values are the corresponding
            word pieces.
            - A dict representing the token symbol, mapping from tokens to IDs.
        """
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_file))

        # Convert word to word piece IDs instead of word piece strings
        # to avoid OOV tokens.
        words_pieces_ids: List[List[int]] = sp.encode(words, out_type=int)

        # Now convert word piece IDs back to word piece strings.
        words_pieces: List[List[str]] = [sp.id_to_piece(ids) for ids in words_pieces_ids]

        lexicon = []
        for word, pieces in zip(words, words_pieces):
            lexicon.append((word, pieces))

        lexicon.append((oov, ["▁", sp.id_to_piece(sp.unk_id())]))

        token2id: Dict[str, int] = {sp.id_to_piece(i): i for i in range(sp.vocab_size())}

        return lexicon, token2id

    @staticmethod
    def add_disambig_symbols(lexicon: Lexicon) -> Tuple[Lexicon, int]:
        """It adds pseudo-token disambiguation symbols #1, #2 and so on
        at the ends of tokens to ensure that all pronunciations are different,
        and that none is a prefix of another.

        See also add_lex_disambig.pl from kaldi.

        Args:
        lexicon:
            It is returned by :func:`read_lexicon`.
        Returns:
        Return a tuple with two elements:

            - The output lexicon with disambiguation symbols
            - The ID of the max disambiguation symbol that appears
            in the lexicon
        """

        # (1) Work out the count of each token-sequence in the
        # lexicon.
        count = defaultdict(int)
        for _, tokens in lexicon:
            count[" ".join(tokens)] += 1

        # (2) For each left sub-sequence of each token-sequence, note down
        # that it exists (for identifying prefixes of longer strings).
        issubseq = defaultdict(int)
        for _, tokens in lexicon:
            tokens = tokens.copy()
            tokens.pop()
            while tokens:
                issubseq[" ".join(tokens)] = 1
                tokens.pop()

        # (3) For each entry in the lexicon:
        # if the token sequence is unique and is not a
        # prefix of another word, no disambig symbol.
        # Else output #1, or #2, #3, ... if the same token-seq
        # has already been assigned a disambig symbol.
        ans = []

        # We start with #1 since #0 has its own purpose
        first_allowed_disambig = 1
        max_disambig = first_allowed_disambig - 1
        last_used_disambig_symbol_of = defaultdict(int)

        for word, tokens in lexicon:
            tokenseq = " ".join(tokens)
            assert tokenseq != ""
            if issubseq[tokenseq] == 0 and count[tokenseq] == 1:
                ans.append((word, tokens))
                continue

            cur_disambig = last_used_disambig_symbol_of[tokenseq]
            if cur_disambig == 0:
                cur_disambig = first_allowed_disambig
            else:
                cur_disambig += 1

            if cur_disambig > max_disambig:
                max_disambig = cur_disambig
            last_used_disambig_symbol_of[tokenseq] = cur_disambig
            tokenseq += f" #{cur_disambig}"
            ans.append((word, tokenseq.split()))
        return ans, max_disambig
    
    @staticmethod
    def write_mapping(filename: str, sym2id: Dict[str, int]) -> None:
        """Write a symbol to ID mapping to a file.

        Note:
        No need to implement `read_mapping` as it can be done
        through :func:`k2.SymbolTable.from_file`.

        Args:
        filename:
            Filename to save the mapping.
        sym2id:
            A dict mapping symbols to IDs.
        Returns:
        Return None.
        """
        with open(filename, "w", encoding="utf-8") as f:
            for sym, i in sym2id.items():
                f.write(f"{sym} {i}\n")
    
    def generate_tokens(self):
        if not (self.lang_dir / "tokens.txt").is_file():
            model_file = self.lang_dir / "bpe.model"
            word_sym_table = k2.SymbolTable.from_file(self.lang_dir / "words.txt")
            words = word_sym_table.symbols
            excluded = ["<eps>", "!SIL", "<SPOKEN_NOISE>", "<UNK>", "#0", "<s>", "</s>"]
            for w in excluded:
                if w in words:
                    words.remove(w)
            lexicon, token_sym_table = PrepareBPE.generate_lexicon(model_file, words, "<UNK>")
            lexicon_disambig, max_disambig = PrepareBPE.add_disambig_symbols(lexicon)
            next_token_id = max(token_sym_table.values()) + 1
            for i in range(max_disambig + 1):
                disambig = f"#{i}"
                assert disambig not in token_sym_table
                token_sym_table[disambig] = next_token_id
                next_token_id += 1
            word_sym_table.add("#0")
            word_sym_table.add("<s>")
            word_sym_table.add("</s>")
            PrepareBPE.write_mapping(self.lang_dir / "tokens.txt", token_sym_table)
            write_lexicon(self.lang_dir / "lexicon.txt", lexicon)
            write_lexicon(self.lang_dir / "lexicon_disambig.txt", lexicon_disambig)


def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config_file, encoding='utf8') as f:
        params = AttributeDict(yaml.safe_load(f))
    prepare_bpe = PrepareBPE(params)
    prepare_bpe.train_bpe_model()
    prepare_bpe.generate_tokens()

if __name__ == '__main__':
    main()