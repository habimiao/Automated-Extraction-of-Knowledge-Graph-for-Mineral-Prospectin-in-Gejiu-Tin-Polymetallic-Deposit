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

# 测试或预测
def predict_from_file(file_path):
    model = RoBERTa_BiLSTM_CRF(roberta_path, len(id2tag),
                               lstm_hidden_dim=lstm_hidden_dim, lstm_dropout_rate=lstm_dropout_rate).to(device)
    model.load_state_dict(torch.load('models/model_epoch_110.pt'))
    model.eval()

    predictions_folder = 'predictions'
    os.makedirs(predictions_folder, exist_ok=True)

    text_save_path = os.path.join(predictions_folder, 'test_texts.text')

    dataset_test = NERDataset(file_path, model.tokenizer, text_save_path=text_save_path)
    dataloader_test = DataLoader(dataset_test, collate_fn=dataset_test.collate_fn, batch_size=batch_size, shuffle=False, drop_last=False)

    all_predictions = []

    with torch.no_grad():
        for i, (input, mask, _) in enumerate(tqdm(dataloader_test, desc='Predict')):
            input, mask = [_.to(device) for _ in (input, mask)]
            y_pred = model(input, mask)
            y_pred = decode_tags_from_ids(y_pred)
            all_predictions.extend(y_pred)

    predictions_path = os.path.join(predictions_folder, 'predictions.txt')
    with open(predictions_path, 'w', encoding='utf-8') as f_pred, open(file_path, 'r', encoding='utf-8') as f_text:
        text_lines = f_text.readlines()
        for prediction, text_line in zip(all_predictions, text_lines):
            text_dict = eval(text_line.strip())
            text = text_dict['text']
            text_tokens = list(text)

            # 如果预测结果的第一个标签为"O"，则将其删除，并将预测结果整体向前平移一个字符位置，最后在结尾添加一个“O”
            if prediction[0] == 'O':
                prediction = prediction[1:]
            prediction.append('O')

            # 输出预测结果
            f_pred.write(f"Text: {text}\nPrediction: {str(prediction)}\n\n")
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

    # 将同一标签的内容合并
    for label in labeled_texts:
        labeled_texts[label] = '，'.join(labeled_texts[label])

    return labeled_texts


# 使用文件进行预测
file_path = 'NER/test.json'
predictions = predict_from_file(file_path)
print(predictions)


