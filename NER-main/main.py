import torch
import wandb
from package.model import RoBERTa_BiLSTM_CRF
from package.dataset import NERDataset, DataLoader, id2tag, decode_tags_from_ids
from package.metrics import Score
from torch.optim import Adam
from tqdm import tqdm
from pprint import pprint
import os

# 初始化wandb
try:
    wandb.init(project='your_project_name')
    wandb_connected = True
except Exception as e:
    print(f"W&B initialization failed: {e}")
    wandb_connected = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
lr = 1e-3
batch_size = 128
accumulation_steps = 1
roberta_path = 'resource/chinese_roberta_wwm_large_ext_pytorch'

lstm_hidden_dim = 768
lstm_dropout_rate = 0.1

if wandb_connected:
    wandb.config.update({
        "learning_rate": lr,
        "batch_size": batch_size,
        "lstm_hidden_dim": lstm_hidden_dim,
        "lstm_dropout_rate": lstm_dropout_rate,
        "roberta_path": roberta_path,
        "accumulation_steps": accumulation_steps,
    })

model = RoBERTa_BiLSTM_CRF(roberta_path, len(id2tag),
                           lstm_hidden_dim=lstm_hidden_dim, lstm_dropout_rate=lstm_dropout_rate).to(device)
model.reset_parameters()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

# 实例化训练集数据集，并将文本保存到 'train_texts.text' 文件中
dataset_train = NERDataset('NER/train.json', model.tokenizer, text_save_path='train_texts.text')
dataloader_train = DataLoader(dataset_train, collate_fn=dataset_train.collate_fn,
                              batch_size=batch_size, shuffle=True, drop_last=True)

# 实例化验证集数据集，并将文本保存到 'dev_texts.text' 文件中
dataset_valid = NERDataset('NER/dev.json', model.tokenizer, text_save_path='dev_texts.text')
dataloader_valid = DataLoader(dataset_valid, collate_fn=dataset_valid.collate_fn,
                              batch_size=batch_size, shuffle=False, drop_last=False)

# 创建保存模型的目录
os.makedirs('models', exist_ok=True)

# 用于保存最后30个epoch模型的列表
last_30_models = []

for epoch in range(120):  # 从500减少到150
    model.train()
    total_loss = 0.0
    with tqdm(desc='Train', total=len(dataloader_train)) as t:
        for i, (input, mask, label) in enumerate(dataloader_train):
            input, mask, label = [_.to(device) for _ in (input, mask, label)]
            loss = model.loss(input, mask, label)
            loss.backward()
            t.update(1)
            t.set_postfix(loss=float(loss))
            total_loss += loss.item()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader_train)
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")  # 输出平均损失
        if wandb_connected:
            wandb.log({"Train Loss": avg_loss}, step=epoch)

        # 保存模型逻辑
        if avg_loss < 10.0 or epoch >= 100:
            model_path = f'models/model_epoch_{epoch}.pt'
            torch.save(model.state_dict(), model_path)
            last_30_models.append(model_path)
            print(f"Model saved at epoch {epoch} with avg_loss {avg_loss}")  # 输出保存模型信息
            if len(last_30_models) > 30:
                removed_model = last_30_models.pop(0)
                os.remove(removed_model)
                print(f"Removed old model: {removed_model}")  # 输出删除模型信息

    model.eval()
    with torch.no_grad():
        score = Score()

        for i, (input, mask, label) in enumerate(tqdm(dataloader_valid, desc='Test')):
            input, mask, label = [_.to(device) for _ in (input, mask, label)]
            y_pred = model(input, mask)

            y_pred = decode_tags_from_ids(y_pred)
            y_true = decode_tags_from_ids(label)

            score.update(y_pred, y_true)

        metrics = score.compute()
        pprint(metrics)

        if wandb_connected:
            if not isinstance(metrics, dict):
                print("metrics:", metrics)
                metrics_dict = {f"metric_{i}": value for i, value in enumerate(metrics)}
                wandb.log(metrics_dict, step=epoch)
            else:
                wandb.log(metrics, step=epoch)

# 在脚本结束时
if wandb_connected:
    wandb.finish()

# 提示是否重新上传数据到W&B
if not wandb_connected:
    retry = input("W&B was not connected during training. Would you like to retry uploading the logs? (yes/no): ")
    if retry.lower() == 'yes':
        try:
            wandb.init(project='your_project_name')
            wandb_connected = True
            if wandb_connected:
                for epoch in range(150):
                    wandb.log({"Train Loss": avg_loss}, step=epoch)
                    if not isinstance(metrics, dict):
                        print("metrics:", metrics)
                        metrics_dict = {f"metric_{i}": value for i, value in enumerate(metrics)}
                        wandb.log(metrics_dict, step=epoch)
                    else:
                        wandb.log(metrics, step=epoch)
                wandb.finish()
        except Exception as e:
            print(f"Retrying W&B upload failed: {e}")
