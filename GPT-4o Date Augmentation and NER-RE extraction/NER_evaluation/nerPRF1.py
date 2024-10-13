import json
import pandas as pd
from collections import defaultdict


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def extract_entities_origin(data):
    result = {}
    for idx, item in enumerate(data):
        text = item['text']
        entities = defaultdict(list)
        for entity in item['entities']:
            label = entity['label']
            start_offset = entity['start_offset']
            end_offset = entity['end_offset']
            text_segment = text[start_offset:end_offset]
            entities[label].append(text_segment)
        result[idx] = entities
    return result


def extract_entities_ner(data):
    result = {}
    for idx, item in enumerate(data):
        entities = defaultdict(list)
        for entity in item['entities']:
            label = entity['label']
            text_segment = entity['text']
            entities[label].append(text_segment)
        result[idx] = entities
    return result


def calculate_metrics(true_entities, pred_entities):
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)

    for key in true_entities:
        if key in pred_entities:
            for label, true_texts in true_entities[key].items():
                pred_texts = pred_entities[key].get(label, [])
                true_texts_set = set(true_texts)
                pred_texts_set = set(pred_texts)

                for true_text in true_texts_set:
                    if any(true_text in pred_text for pred_text in pred_texts_set):
                        true_positive[label] += 1
                    else:
                        false_negative[label] += 1

                for pred_text in pred_texts_set:
                    if not any(true_text in pred_text for true_text in true_texts_set):
                        false_positive[label] += 1

    metrics = {}
    for label in set(true_positive.keys()).union(false_positive.keys()).union(false_negative.keys()):
        precision = true_positive[label] / (true_positive[label] + false_positive[label]) if (true_positive[label] +
                                                                                              false_positive[
                                                                                                  label]) > 0 else 0
        recall = true_positive[label] / (true_positive[label] + false_negative[label]) if (true_positive[label] +
                                                                                           false_negative[
                                                                                               label]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[label] = (precision, recall, f1)

    return metrics


def generate_results_table(metrics):
    labels = [
        "本体", "矿床知识本体", "副标题", "数据参数其他", "图表", "属性：数值(比例)等", "品位", "储量",
        "地图术语、地质术语", "数据", "网距", "参数", "分量", "编号；代码", "数据统计术语", "异常",
        "重力异常范围", "颜色", "化学参数，术语", "序列", "公式", "温度范围", "区域与位置",
        "地理位置、方位、方向等", "矿区位置", "矿区", "区域，矿集区", "地层", "岩层", "序列位置",
        "矿床基础知识", "矿床", "矿段", "矿带、岩石带等", "矿体分布", "矿床类别", "矿体", "矿脉",
        "矿床谱系、矿床系列", "矿体形态", "矿体部位", "矿物矿石相关", "矿物", "矿物类型",
        "矿物形态", "矿石结构/构造", "条件特点", "地物化条件/特征", "特点", "趋势、数据分布、程度、顺序",
        "形成原因", "作用方式", "作用条件", "作用过程", "数据条件、部位", "岩石岩浆相关", "岩石",
        "岩石密度", "岩体", "岩体部位", "岩石类型", "岩石/矿物微观结构特征", "岩石结构", "岩浆",
        "岩浆作用", "围岩相关", "蚀变作用", "围岩", "年代相关", "地质年代", "年代地层",
        "探测与分析方法", "数据统计方法", "同位素方法", "方法、程序、原理", "实验人物部门设备",
        "部门、人员、研究所、实验室", "实验处理（样品）", "仪器设备", "地质结构及活动", "地质构造/结构",
        "地质活动", "构造部位", "构造活动", "成/容/控矿构造", "构造特点", "构造形态", "地球化学相关",
        "矿化作用", "化学元素", "化学组分", "矿化阶段", "矿化状态", "分带序列", "成矿相关",
        "成矿条件特点", "成矿阶段、成矿期", "成矿流体"
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
def count_entities_frequency(entities):
    """
    统计每个子标签在原始标注中的频数
    """
    frequencies = defaultdict(int)
    for item in entities.values():
        for label, texts in item.items():
            frequencies[label] += len(texts)
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

# 读取文件
data_origin = load_json('updated_output_or_file.jsonl')
output_ner = load_json('updated_output_ner_file.jsonl')

# 提取entities
true_entities = extract_entities_origin(data_origin)
print(true_entities)
pred_entities = extract_entities_ner(output_ner)
print(pred_entities)
# 计算每个label的指标
metrics = calculate_metrics(true_entities, pred_entities)
# 统计每个子标签在原始标注中的频数
frequencies = count_entities_frequency(true_entities)
# 定义标签组
label_groups = {
    "本体": ["矿床知识本体", "副标题"],
    "数据参数其他": ["图表", "属性：数值(比例)等", "品位", "储量", "地图术语、地质术语", "数据", "网距", "参数", "分量", "编号；代码", "数据统计术语", "异常", "重力异常范围", "颜色", "化学参数，术语", "序列", "公式", "温度范围"],
    "区域与位置": ["地理位置、方位、方向等", "矿区位置", "矿区", "区域，矿集区", "地层", "岩层", "序列位置"],
    "矿床基础知识": ["矿床", "矿段", "矿带、岩石带等", "矿体分布", "矿床类别", "矿体", "矿脉", "矿床谱系、矿床系列", "矿体形态", "矿体部位"],
    "矿物矿石相关": ["矿物", "矿物类型", "矿物形态", "矿石结构/构造"],
    "条件特点": ["地物化条件/特征", "特点", "趋势、数据分布、程度、顺序", "形成原因", "作用方式", "作用条件", "作用过程", "数据条件、部位"],
    "岩石岩浆相关": ["岩石", "岩石密度", "岩体", "岩体部位", "岩石类型", "岩石/矿物微观结构特征", "岩石结构", "岩浆", "岩浆作用"],
    "围岩相关": ["蚀变作用", "围岩"],
    "年代相关": ["地质年代", "年代地层"],
    "探测与分析方法": ["数据统计方法", "同位素方法", "方法、程序、原理"],
    "实验人物部门设备": ["部门、人员、研究所、实验室", "实验处理（样品）", "仪器设备"],
    "地质结构及活动": ["地质构造/结构", "地质活动", "构造部位", "构造活动", "成/容/控矿构造", "构造特点", "构造形态"],
    "地球化学相关": ["矿化作用", "化学元素", "化学组分", "矿化阶段", "矿化状态", "分带序列"],
    "成矿相关": ["成矿条件特点", "成矿阶段、成矿期", "成矿流体"]
}

# 聚合子标签的指标到父标签
# aggregate_metrics(metrics, label_groups)

# 使用加权聚合子标签的指标到父标签
aggregate_metrics_with_weighting(metrics, label_groups, frequencies)
# 生成结果表格
results_df = generate_results_table(metrics)

# 输出结果表格
results_df.to_csv('f1_results.csv', index=False)
print("F1 scores have been saved to f1_results.csv")