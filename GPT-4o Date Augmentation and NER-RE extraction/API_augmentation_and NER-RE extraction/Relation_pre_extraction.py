import json

# 文件路径
input_file_path = r'.\data_origin.json'
output_file_path = r'.\re-op-origin.json'


def process_line(line):
    data = json.loads(line)
    # 过滤掉'relations'为空列表的数据
    if not data.get('relations') or data['relations'] == []:
        return None

    # 处理'entities'中的每个实体，添加'text'字段
    entity_dict = {}
    for entity in data['entities']:
        start = entity['start_offset']
        end = entity['end_offset']
        entity_text = data['text'][start:end]
        entity['text'] = entity_text
        entity_dict[entity['id']] = entity_text

    # 处理'relations'中的每个关系，添加'start_text'和'end_text'字段
    for i, relation in enumerate(data['relations']):
        from_text = entity_dict.get(relation['from_id'], "")
        to_text = entity_dict.get(relation['to_id'], "")
        # 创建新的relation字典，确保字段顺序正确
        data['relations'][i] = {
            "id": relation["id"],
            "from_id": relation["from_id"],
            "start_text": from_text,
            "to_id": relation["to_id"],
            "end_text": to_text,
            "type": relation["type"]
        }
    return data


def main():
    with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w',
                                                                      encoding='utf-8') as outfile:
        for line in infile:
            processed_data = process_line(line)
            if processed_data:
                json.dump(processed_data, outfile, ensure_ascii=False)
                outfile.write('\n')

if __name__ == "__main__":
    main()