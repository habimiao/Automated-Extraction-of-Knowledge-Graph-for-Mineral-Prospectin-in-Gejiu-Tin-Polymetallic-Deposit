# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import read_clue_json

tag2id = {'O': 0,
          'B-本体': 1, 'I-本体': 2,
          'B-矿床': 3, 'I-矿床': 4,
          'B-矿床类别': 5, 'I-矿床类别': 6,
          'B-矿床谱系、矿床系列': 7, 'I-矿床谱系、矿床系列': 8,

          'B-岩层': 9, 'I-岩层': 10,
          'B-地层': 11, 'I-地层': 12,
          'B-区域，矿集区': 13, 'I-区域，矿集区': 14,
          'B-矿区': 15, 'I-矿区': 16,
          'B-图表': 17, 'I-图表': 18,

          'B-矿体': 19, 'I-矿体': 20,
          'B-矿体形态': 21, 'I-矿体形态': 22,
          'B-矿体部位': 23, 'I-矿体部位': 24,
          'B-矿带、岩石带等': 25, 'I-矿带、岩石带等': 26,
          'B-矿脉': 27, 'I-矿脉': 28,
          'B-成矿阶段、成矿期': 29, 'I-成矿阶段、成矿期': 30,
          'B-成矿流体': 31, 'I-成矿流体': 32,
          'B-岩石': 33, 'I-岩石': 34,
          'B-岩体': 35, 'I-岩体': 36,
          'B-岩体部位': 37, 'I-岩体部位': 38,
          'B-矿段': 39, 'I-矿段': 40,
          'B-岩浆': 41, 'I-岩浆': 42,
          'B-岩浆作用': 43, 'I-岩浆作用': 44,
          'B-岩石类型': 45, 'I-岩石类型': 46,
          'B-岩石结构': 47, 'I-岩石结构': 48,
          'B-岩石/矿物微观结构特征': 49, 'I-岩石/矿物微观结构特征': 50,
          'B-矿物': 51, 'I-矿物': 52,
          'B-矿物形态': 53, 'I-矿物形态': 54,
          'B-矿物类型': 55, 'I-矿物类型': 56,
          'B-矿石结构/构造': 57, 'I-矿石结构/构造': 58,
          'B-地质构造/结构': 59, 'I-地质构造/结构': 60,
          'B-成/容/控矿构造': 61, 'I-成/容/控矿构造': 62,
          'B-构造形态': 63, 'I-构造形态': 64,
          'B-构造部位': 65, 'I-构造部位': 66,
          'B-构造活动': 67, 'I-构造活动': 68,
          'B-地质活动': 69, 'I-地质活动': 70,
          'B-围岩': 71, 'I-围岩': 72,
          'B-蚀变作用': 73, 'I-蚀变作用': 74,
          'B-地质年代': 75, 'I-地质年代': 76,
          'B-年代地层': 77, 'I-年代地层': 78,
          'B-矿化作用': 79, 'I-矿化作用': 80,
          'B-矿化状态': 81, 'I-矿化状态': 82,
          'B-矿化阶段': 83, 'I-矿化阶段': 84,

          'B-地球化学相关': 85, 'I-地球化学相关': 86,
          'B-探测与分析方法': 87, 'I-探测与分析方法': 88,
          'B-实验人物部门设备': 89, 'I-实验人物部门设备': 90,
          'B-区域与位置': 91, 'I-区域与位置': 92,
          'B-数据参数其他': 93, 'I-数据参数其他': 94,
          'B-条件特点': 95, 'I-条件特点': 96,

          'B-矿体分布': 97, 'I-矿体分布': 98,
          }

id2tag = {
    0: 'O',
    1: 'B-本体', 2: 'I-本体',
    3: 'B-矿床', 4: 'I-矿床',
    5: 'B-矿床类别', 6: 'I-矿床类别',
    7: 'B-矿床谱系、矿床系列', 8: 'I-矿床谱系、矿床系列',

    9: 'B-岩层', 10: 'I-岩层',
    11: 'B-地层', 12: 'I-地层',
    13: 'B-区域，矿集区', 14: 'I-区域，矿集区',
    15: 'B-矿区', 16: 'I-矿区',
    17: 'B-图表', 18: 'I-图表',

    19: 'B-矿体', 20: 'I-矿体',
    21: 'B-矿体形态', 22: 'I-矿体形态',
    23: 'B-矿体部位', 24: 'I-矿体部位',
    25: 'B-矿带、岩石带等', 26: 'I-矿带、岩石带等',
    27: 'B-矿脉', 28: 'I-矿脉',
    29: 'B-成矿阶段、成矿期', 30: 'I-成矿阶段、成矿期',
    31: 'B-成矿流体', 32: 'I-成矿流体',
    33: 'B-岩石', 34: 'I-岩石',
    35: 'B-岩体', 36: 'I-岩体',
    37: 'B-岩体部位', 38: 'I-岩体部位',
    39: 'B-矿段', 40: 'I-矿段',
    41: 'B-岩浆', 42: 'I-岩浆',
    43: 'B-岩浆作用', 44: 'I-岩浆作用',
    45: 'B-岩石类型', 46: 'I-岩石类型',
    47: 'B-岩石结构', 48: 'I-岩石结构',
    49: 'B-岩石/矿物微观结构特征', 50: 'I-岩石/矿物微观结构特征',
    51: 'B-矿物', 52: 'I-矿物',
    53: 'B-矿物形态', 54: 'I-矿物形态',
    55: 'B-矿物类型', 56: 'I-矿物类型',
    57: 'B-矿石结构/构造', 58: 'I-矿石结构/构造',
    59: 'B-地质构造/结构', 60: 'I-地质构造/结构',
    61: 'B-成/容/控矿构造', 62: 'I-成/容/控矿构造',
    63: 'B-构造形态', 64: 'I-构造形态',
    65: 'B-构造部位', 66: 'I-构造部位',
    67: 'B-构造活动', 68: 'I-构造活动',
    69: 'B-地质活动', 70: 'I-地质活动',
    71: 'B-围岩', 72: 'I-围岩',
    73: 'B-蚀变作用', 74: 'I-蚀变作用',
    75: 'B-地质年代', 76: 'I-地质年代',
    77: 'B-年代地层', 78: 'I-年代地层',
    79: 'B-矿化作用', 80: 'I-矿化作用',
    81: 'B-矿化状态', 82: 'I-矿化状态',
    83: 'B-矿化阶段', 84: 'I-矿化阶段',
    85: 'B-地球化学相关', 86: 'I-地球化学相关',
    87: 'B-探测与分析方法', 88: 'I-探测与分析方法',
    89: 'B-实验人物部门设备', 90: 'I-实验人物部门设备',
    91: 'B-区域与位置', 92: 'I-区域与位置',
    93: 'B-数据参数其他', 94: 'I-数据参数其他',
    95: 'B-条件特点', 96: 'I-条件特点',
    97: 'B-矿体分布', 98: 'I-矿体分布',
          }


def decode_tags_from_ids(batch_ids):
    batch_tags = []
    for ids in batch_ids:
        sequence_tags = []
        for id in ids:
            sequence_tags.append(id2tag[int(id)])
        batch_tags.append(sequence_tags)
    return batch_tags


class NERDataset(Dataset):
    """Pytorch Dataset
    """

    def __init__(self, path_to_clue, tokenizer, text_save_path):
        self.data = read_clue_json(path_to_clue)
        self.tokenizer = tokenizer
        self.text_save_path = text_save_path
        self.save_texts()

    def save_texts(self):
        with open(self.text_save_path, 'w', encoding='utf-8') as f:
            for pkg in self.data:
                text = pkg['text']
                labels = pkg["label"]
                for char, label in zip(text, labels):
                    f.write(f"{char} {label}\n")
                f.write("\n")

    def collate_fn(self, batch):
        """collate_fn for 'torch.utils.data.DataLoader'
        """
        texts, labels = list(zip(*[[item[0], item[1]] for item in batch]))
        token = self.tokenizer(list(texts), padding=False, return_offsets_mapping=True)

        # align the label
        # Bert mat split a word 'AA' into 'A' and '##A'
        labels = [self._align_label(offset, label) for offset, label in zip(token['offset_mapping'], labels)]
        token = self.tokenizer.pad(token, padding=True, return_attention_mask=True)

        return torch.LongTensor(token['input_ids']), torch.ByteTensor(token['attention_mask']), self._pad(labels)

    @staticmethod
    def _align_label(offset, label):

        label_align = []
        for i, (start, end) in enumerate(offset):

            if start == end:
                label_align.append(tag2id['O'])
            else:
                # 1-N or N-1, default to use first original label as final label
                if i > 0 and offset[i - 1] == (start, end):
                    label_align.append(label[start:end][0].replace('B', 'O', 1))
                else:
                    label_align.append(label[start:end][0])
        return label_align

    @staticmethod
    def _pad(labels):
        max_len = max([len(label) for label in labels])
        labels = [(label + [tag2id['O']] * (max_len - len(label))) for label in labels]
        return torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pkg = self.data[index]

        text = pkg['text']
        label = [tag2id[tag] for tag in pkg["label"]]

        return text, label

