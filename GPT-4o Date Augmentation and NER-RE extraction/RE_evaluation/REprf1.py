import json
import pandas as pd
from collections import defaultdict

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def extract_relations_origin(data):
    result = {}
    for idx, item in enumerate(data):
        relations = defaultdict(list)
        for relation in item['relations']:
            relations[relation['type']].append((relation['start_text'], relation['end_text']))
        result[idx] = relations
    return result

def extract_relations_ner(data):
    result = {}
    for idx, item in enumerate(data):
        relations = defaultdict(list)
        for relation in item['relations']:
            relations[relation['type']].append((relation['start_text'], relation['end_text']))
        result[idx] = relations
    return result

def extract_relations_ner_no_order(data):
    result = {}
    for idx, item in enumerate(data):
        relations = defaultdict(list)
        for relation in item['relations']:
            start_text, end_text = sorted([relation['start_text'], relation['end_text']])
            relations[relation['type']].append((start_text, end_text))
        result[idx] = relations
    return result

def calculate_metrics(true_relations, pred_relations):
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)

    for key in true_relations:
        if key in pred_relations:
            for label, true_pairs in true_relations[key].items():
                pred_pairs = pred_relations[key].get(label, [])
                true_pairs_set = set(true_pairs)
                pred_pairs_set = set(pred_pairs)

                for true_pair in true_pairs_set:
                    if true_pair in pred_pairs_set:
                        true_positive[label] += 1
                    else:
                        false_negative[label] += 1

                for pred_pair in pred_pairs_set:
                    if pred_pair not in true_pairs_set:
                        false_positive[label] += 1

    metrics = {}
    for label in set(true_positive.keys()).union(false_positive.keys()).union(false_negative.keys()):
        precision = true_positive[label] / (true_positive[label] + false_positive[label]) if (true_positive[label] + false_positive[label]) > 0 else 0
        recall = true_positive[label] / (true_positive[label] + false_negative[label]) if (true_positive[label] + false_negative[label]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[label] = (precision, recall, f1)

    return metrics

def generate_results_table(metrics):
    labels = [
        "量度与比较", "约", "从", "到向", "平均值", "一致", "重要", "占", "比较", "主要", "小于", "只有", "次要", "相似", "大于", "超出", "加权",
        "组成构成属性关系", "有", "为", "包含", "的", "具有", "和", "或", "与", "以及", "分为", "少数", "多数", "部分", "其余",
        "位置与地理关系", "在", "包围", "延伸", "以界", "接临相连", "分布于", "靠近", "之间", "产于", "集中于", "夹", "相伴", "展开", "围绕", "远离", "穿切", "沿着", "中心", "距离", "平行于", "相遇", "穿插", "叠加在",
        "时间次序发展关系", "转到", "随着", "之前", "之后", "持续", "晚于",
        "过程与机制", "发育", "用", "对", "充填", "形成", "蚀变", "矿化", "出露", "通过", "富集", "隐伏", "侵入", "过渡", "计算", "迁移", "赋存", "共生", "伴生", "产出", "拟合",
        "条件假设与逻辑推理", "相关", "来源于", "可能", "成因", "刻画了，反映了", "控制", "说明", "预测推测", "受控于", "根据", "为例", "无", "可见", "利于", "认为", "影响", "提供", "导致", "以条件", "找寻", "结果表明", "总结", "圈定", "制作"
    ]

    rows = []
    for label in labels:
        if label in metrics:
            precision, recall, f1 = metrics[label]
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0
        rows.append([label, precision, recall, f1])
    df = pd.DataFrame(rows, columns=["Label", "Precision", "Recall", "F1 Score"])
    return df

def aggregate_metrics(metrics, label_groups):
    for group, sublabels in label_groups.items():
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        count = 0

        for sublabel in sublabels:
            if sublabel in metrics:
                precision, recall, f1 = metrics[sublabel]
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                count += 1

        if count > 0:
            average_precision = total_precision / count
            average_recall = total_recall / count
            average_f1 = total_f1 / count
            metrics[group] = (average_precision, average_recall, average_f1)
        else:
            metrics[group] = (0.0, 0.0, 0.0)
def count_relations_frequency(relations):
    """
    统计每个关系标签在原始标注中的频数
    """
    frequencies = defaultdict(int)
    for item in relations.values():
        for label, pairs in item.items():
            frequencies[label] += len(pairs)
    return frequencies

def aggregate_metrics_with_weighting(metrics, label_groups, frequencies):
    """
    使用子标签的频数进行加权计算父标签的指标
    """
    for group, sublabels in label_groups.items():
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_weight = 0

        for sublabel in sublabels:
            if sublabel in metrics and sublabel in frequencies:
                precision, recall, f1 = metrics[sublabel]
                weight = frequencies[sublabel]
                total_precision += precision * weight
                total_recall += recall * weight
                total_f1 += f1 * weight
                total_weight += weight

        if total_weight > 0:
            average_precision = total_precision / total_weight
            average_recall = total_recall / total_weight
            average_f1 = total_f1 / total_weight
            metrics[group] = (average_precision, average_recall, average_f1)
        else:
            metrics[group] = (0.0, 0.0, 0.0)

# 定义标签组
label_groups = {
    "量度与比较": ["约", "从", "到向", "平均值", "一致", "重要", "占", "比较", "主要", "小于", "只有", "次要", "相似", "大于", "超出", "加权"],
    "组成构成属性关系": ["有", "为", "包含", "的", "具有", "和", "或", "与", "以及", "分为", "少数", "多数", "部分", "其余"],
    "位置与地理关系": ["在", "包围", "延伸", "以界", "接临相连", "分布于", "靠近", "之间", "产于", "集中于", "夹", "相伴", "展开", "围绕", "远离", "穿切", "沿着", "中心", "距离", "平行于", "相遇", "穿插", "叠加在"],
    "时间次序发展关系": ["转到", "随着", "之前", "之后", "持续", "晚于"],
    "过程与机制": ["发育", "用", "对", "充填", "形成", "蚀变", "矿化", "出露", "通过", "富集", "隐伏", "侵入", "过渡", "计算", "迁移", "赋存", "共生", "伴生", "产出", "拟合"],
    "条件假设与逻辑推理": ["相关", "来源于", "可能", "成因", "刻画了，反映了", "控制", "说明", "预测推测", "受控于", "根据", "为例", "无", "可见", "利于", "认为", "影响", "提供", "导致", "以条件", "找寻", "结果表明", "总结", "圈定", "制作"]
}

# 读取文件
data_origin = load_json('out-re-reset-origin.json')
output_ner = load_json('out-re-reset.jsonl')

# 提取关系
true_relations = extract_relations_origin(data_origin)
pred_relations = extract_relations_ner(output_ner)
pred_relations_no_order = extract_relations_ner_no_order(output_ner)

# 统计每个标签的频数
frequencies = count_relations_frequency(true_relations)

# 计算每个label的指标（考虑先后顺序）
metrics_ordered = calculate_metrics(true_relations, pred_relations)

# 使用加权聚合子标签的指标到父标签
aggregate_metrics_with_weighting(metrics_ordered, label_groups, frequencies)

# 生成结果表格（考虑先后顺序）
results_df_ordered = generate_results_table(metrics_ordered)
results_df_ordered.to_csv('f1_results_ordered.csv', index=False)
print("F1 scores (ordered) have been saved to f1_results_ordered.csv")

# 计算每个label的指标（不考虑先后顺序）
metrics_unordered = calculate_metrics(true_relations, pred_relations_no_order)

# 使用加权聚合子标签的指标到父标签
aggregate_metrics_with_weighting(metrics_unordered, label_groups, frequencies)

# 生成结果表格（不考虑先后顺序）
# results_df_unordered = generate_results_table(metrics_unordered)
# results_df_unordered.to_csv('f1_results_unordered.csv', index=False)
# print("F1 scores (unordered) have been saved to f1_results_unordered.csv")
