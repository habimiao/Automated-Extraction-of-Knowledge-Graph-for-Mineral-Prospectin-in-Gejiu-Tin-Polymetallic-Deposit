import torch
import os
from tqdm import tqdm
from package.model import RoBERTa_BiLSTM_CRF
from package.dataset import NERDataset, DataLoader, id2tag, decode_tags_from_ids
from package.metrics import Score
from torch.optim import Adam
from pprint import pprint

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
lr = 1e-3
batch_size = 128
accumulation_steps = 1
roberta_path = 'resource/chinese_roberta_wwm_large_ext_pytorch'

lstm_hidden_dim = 768
lstm_dropout_rate = 0.1

def predict_single_sentence(sentence):
    model = RoBERTa_BiLSTM_CRF(roberta_path, len(id2tag),
                               lstm_hidden_dim=lstm_hidden_dim, lstm_dropout_rate=lstm_dropout_rate).to(device)
    model.load_state_dict(torch.load('models/model_epoch_86.pt'))
    model.eval()

    predictions_folder = 'predictions'
    os.makedirs(predictions_folder, exist_ok=True)
    prediction_file_path = os.path.join(predictions_folder, 'prediction-t.txt')

    # 创建临时文件用于数据集初始化
    temp_file_path = 'temp_test.json'
    text_save_path = 'temp_texts.text'
    with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
        temp_file.write(f'{{"text": "{sentence}"}}\n')

    dataset_test = NERDataset(temp_file_path, model.tokenizer, text_save_path=text_save_path)
    dataloader_test = DataLoader(dataset_test, collate_fn=dataset_test.collate_fn, batch_size=1, shuffle=False, drop_last=False)

    all_predictions = []

    with torch.no_grad():
        for input, mask, _ in tqdm(dataloader_test, desc='Predict'):
            input, mask = [_.to(device) for _ in (input, mask)]
            y_pred = model(input, mask)
            y_pred = decode_tags_from_ids(y_pred)
            all_predictions.extend(y_pred)

    # 删除临时文件
    os.remove(temp_file_path)
    os.remove(text_save_path)

    with open(prediction_file_path, 'w', encoding='utf-8') as f_pred:
        text_tokens = list(sentence)

        # 获取预测结果并修正标签
        prediction = all_predictions[0]
        if prediction[0] == 'O':
            prediction = prediction[1:]
        prediction.append('O')

        # 输出预测结果
        f_pred.write(f"Text: {sentence}\nPrediction: {str(prediction)}\n\n")
        for token, label in zip(text_tokens, prediction):
            f_pred.write(f"{token} {label}\n")
        f_pred.write("\n")

        # 提取标注的文本
        labeled_texts = extract_labeled_texts(text_tokens, prediction)
        for label, content in labeled_texts.items():
            f_pred.write(f"{label}: {content}\n")
        f_pred.write("\n")

    return all_predictions

def extract_labeled_texts(tokens, labels):
    labeled_texts = {}
    current_label = None
    current_text = []

    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            if current_label and current_text:
                if current_label in labeled_texts:
                    labeled_texts[current_label].append(''.join(current_text))
                else:
                    labeled_texts[current_label] = [''.join(current_text)]
            current_label = label[2:]
            current_text = [token]
        elif label.startswith('I-') and current_label:
            current_text.append(token)
        else:
            if current_label and current_text:
                if current_label in labeled_texts:
                    labeled_texts[current_label].append(''.join(current_text))
                else:
                    labeled_texts[current_label] = [''.join(current_text)]
            current_label = None
            current_text = []

    if current_label and current_text:
        if current_label in labeled_texts:
            labeled_texts[current_label].append(''.join(current_text))
        else:
            labeled_texts[current_label] = [''.join(current_text)]

    for label in labeled_texts:
        labeled_texts[label] = '，'.join(labeled_texts[label])

    return labeled_texts
# 使用单句进行预测
# sentence = '矿体主要位于花岗岩体和个旧组卡房段第五层大理岩接触蚀变带内蚀变带(蚀变花岗岩中),矿体最深可达地下1000多米，矿体延伸可达400m,最厚可达40m。'
sentence = '矽卡岩型硫化物矿主要赋存在花岗岩和大理岩的接触带上，然而矿体形态多受花岗岩的空间展布形态控制，多呈皮壳状、透镜状、似层状等'
predictions = predict_single_sentence(sentence)
print(predictions)



