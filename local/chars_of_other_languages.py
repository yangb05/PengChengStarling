from pathlib import Path

import pythainlp
import nagisa
from icefall.utils import tokenize_by_CJK_char


# 定义每种语言的字符范围
language_ranges = {
    'Chinese': [
        (0x4E00, 0x9FFF),  # 常用汉字（CJK Unified Ideographs）
        (0x3400, 0x4DBF),  # 扩展A区汉字
        (0x20000, 0x2A6DF),  # 扩展B区汉字
        (0x2A700, 0x2B73F),  # 扩展C区汉字
        (0x2B740, 0x2B81F),  # 扩展D区汉字
        (0x2B820, 0x2CEAF),  # 扩展E区汉字
        (0xF900, 0xFAFF),   # 兼容汉字
        (0x0030, 0x0039) # 基本阿拉伯数字 0-9
    ],
    'English': [
        (0x0041, 0x005A),  # 大写拉丁字母 A-Z
        (0x0061, 0x007A),  # 小写拉丁字母 a-z
        (0x0030, 0x0039)   # 基本阿拉伯数字 0-9
    ],
    'Vietnamese': [
        (0x0041, 0x005A),  # A-Z
        (0x0061, 0x007A),  # a-z
        (0x00C0, 0x00FF),  # À-ÿ
        (0x0102, 0x0102),  # Ă
        (0x0103, 0x0103),  # ă
        (0x0110, 0x0110),  # Đ
        (0x0111, 0x0111),  # đ
        (0x0128, 0x0128),  # Ĩ
        (0x0129, 0x0129),  # ĩ
        (0x0168, 0x0168),  # Ũ
        (0x0169, 0x0169),  # ũ
        (0x01A0, 0x01A0),  # Ơ
        (0x01A1, 0x01A1),  # ơ
        (0x01AF, 0x01AF),  # Ư
        (0x01B0, 0x01B0),  # ư
        (0x1EA0, 0x1EF9),  # Ạ-ỹ
        (0x0030, 0x0039) # 基本阿拉伯数字 0-9
    ],
    'Russian': [
        (0x0400, 0x04FF),  # 基本西里尔字母
        (0x0500, 0x052F),  # 扩展西里尔字母
        (0x2DE0, 0x2DFF),  # 西里尔字母扩展附加字符
        (0xA640, 0xA69F),  # 西里尔字母扩展-B
        (0x0030, 0x0039) # 基本阿拉伯数字 0-9
    ],
    'Japanese': 
        [
        (0x3040, 0x309F),  # 平假名
        (0x30A0, 0x30FF),  # 片假名
        (0x31F0, 0x31FF),  # 片假名扩展
        (0xFF66, 0xFF9F),  # 半角片假名
        (0x4E00, 0x9FFF),  # 常用汉字（CJK Unified Ideographs）
        (0x3400, 0x4DBF),  # 扩展A区汉字
        (0xF900, 0xFAFF),   # 兼容汉字
        (0x0030, 0x0039) # 基本阿拉伯数字 0-9
    ],
    'Arabic': [
        (0x0600, 0x06FF),  # 基本阿拉伯语字符和符号
        (0x0750, 0x077F),  # 补充阿拉伯语字符
        (0x08A0, 0x08FF),  # 额外的阿拉伯语扩展字符
        (0xFB50, 0xFDFF),  # 阿拉伯语展示形式-A
        (0xFE70, 0xFEFF),  # 阿拉伯语展示形式-B
        (0x1EE00, 0x1EEFF), # 阿拉伯数学字母符号
        (0x0030, 0x0039) # 基本阿拉伯数字 0-9
    ],
    'Thai': [
        (0x0E00, 0x0E7F),  # 泰文字符
        (0x0030, 0x0039) # 基本阿拉伯数字 0-9
    ],
    'Indonesian': [
        (0x0041, 0x005A),  # A-Z
        (0x0061, 0x007A),  # a-z
        (0x00C9, 0x00C9),  # É (é大写形式)
        (0x00E9, 0x00E9),  # é
        (0x00D3, 0x00D3),  # Ó (ó大写形式)
        (0x00F3, 0x00F3),   # ó
        (0x0030, 0x0039) # 基本阿拉伯数字 0-9
    ],
}

hyp=['要跟着我说你就РОW', 'AND', 'IT', 'SAYS', 'SOMETHING', 'LIKE', 'WEIRD', 'ON', 'IT']


def contain_other_languages(language, s):
    language_range = language_ranges[language]
    
    for char in s:
        char_code = ord(char)
        in_language_range = False
        for start, end in language_range:
            if start <= char_code <= end:
                in_language_range = True
                break
        
        # 如果字符不在任何字符范围内，则包含其他语言字符
        if not in_language_range:
            return True
    
    return False

    
def count_other_language_words(language, recogs_file):
    hyps = []
    with open(recogs_file, 'r') as f:
        for idx,line in enumerate(f.readlines()):
            if idx % 2 != 0:
                hyp = eval(line.strip().split('\t', 1)[1].split('=', 1)[1])
                if len(hyp) != 0:
                    if language == 'Chinese':
                        hyp = tokenize_by_CJK_char(hyp[0])
                        hyp = hyp.split()
                    if language == 'Thai':
                        hyp = pythainlp.word_tokenize(hyp[0], keep_whitespace=False)
                    if language == 'Japanese':
                        hyp = nagisa.tagging(hyp[0])
                        hyp = hyp.words
                    hyps += hyp
    count = 0
    for word in hyps:
        if contain_other_languages(language, word):
            count += 1
    rate = count / len(hyps) * 100
    print(f'language: {language} total words: {len(hyps)} other language words: {count} other language rate: {rate:.2f}%')
    

def get_results(recogs_dir, epoch_avg):
    recogs_files = []
    for file in recogs_dir.iterdir():
        if file.name.startswith('recogs') and epoch_avg in file.name:
            recogs_files.append(file)
    # assert len(recogs_files) == 8
    # get language
    print(f'ckpt: {epoch_avg}')
    for recogs_file in recogs_files:
        if 'gigaspeech2-id' in recogs_file.name:   
            count_other_language_words('Indonesian', recogs_file)
        elif 'gigaspeech2-th' in recogs_file.name:
            count_other_language_words('Thai', recogs_file)
        elif 'gigaspeech2-vi' in recogs_file.name:
            count_other_language_words('Vietnamese', recogs_file)
        elif 'gigaspeech' in recogs_file.name:
            count_other_language_words('English', recogs_file)
        elif 'mgb2' in recogs_file.name:
            count_other_language_words('Arabic', recogs_file)
        elif 'reazonspeech' in recogs_file.name:
            count_other_language_words('Japanese', recogs_file)
        elif 'ru-datatang' in recogs_file.name:
            count_other_language_words('Russian', recogs_file)
        elif 'wenetspeech' in recogs_file.name:
            count_other_language_words('Chinese', recogs_file)
        else:
            print(f'Wrong recogs file: {recogs_file}')


if __name__ == '__main__':
    recogs_dir = Path('/mgData1/yangb/icefall/egs/omini/ASR/zipformer/multilingual_8asr_16000h_280M_online_langtag/streaming/greedy_search')
    epoch_avg = 'epoch-20-avg-10'
    get_results(recogs_dir, epoch_avg)
           