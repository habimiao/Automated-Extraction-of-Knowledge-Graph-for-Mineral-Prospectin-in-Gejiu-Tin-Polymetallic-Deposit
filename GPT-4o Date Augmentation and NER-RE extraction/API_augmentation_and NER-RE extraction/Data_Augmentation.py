
import time
from openai import OpenAI
api_key = "" #your open API key

client = OpenAI(api_key=api_key)


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return data


def save_answer(answer, output_file):
    with open(output_file, 'a', encoding='utf-8') as file:
        file.write(answer + '\n')


def process_data(data, start_line, batch_size=1):
    for i in range(start_line, len(data), batch_size):
        messages = []
        for j in range(i, min(i + batch_size, len(data))):
            json_content = data[j].strip()
            print( json_content)
            messages.append({"role": "user", "content": json_content})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                         {"role": "system",
                          "content": "现在请你扮演地质领域锡矿相关找矿专家，对下面给出的json数据进行数据增强。要求：1.采用不影响剧中实体与关系原意的前提下对句子部分内容进行增加、删除、替换操作。注意：“增加”，不能增加原因“因此”、“于是”...以及总结相关词语如“综上可知”、“由上图可得”等相关词语。“删除”，请不要删除原句中与relation抽取相关的词语，删除过程中不要改变原有的entity词语的意义。“替换”，可以对relation抽取的句中关系词进行替换，亦可以对抽取的实体（entity）进行部分不改变原有句意的替换。2.将增强的数据按照与元素据一样的格式进行输出。3.请以代码生成的方式严格执行；对于entities中id、start_offset、end_offset与relations中的from_id、to_id请注意对照关系，relations中的id为序列。其中的id都为序列数字。注意entities中id、start_offset、end_offset经过增删替换后的具体数值更新。请尽量与原提取的entity内容一致，请在生成的过程中进行检查。4.对于输入的一条数据，请尽可能输出数据增强到5条数据。example：输入：{\"id\": 1, \"text\": \"卡房段是高松矿床主要的矿化段。\", \"entities\": [{\"id\": 1, \"label\": \"矿段\", \"start_offset\": 0, \"end_offset\": 3,\"text\":\"卡房段\"}, {\"id\": 2, \"label\": \"矿床实例\", \"start_offset\": 4, \"end_offset\": 8,\"text\":\"高松矿床\"}], \"relations\": [{\"id\": 1, \"from_id\": 1,\"start_text\":\"卡房段\",to_id\": 2, \"end_text\":\"高松矿床\",\"type\": \"主要\"}, {\"id\": 2,\"from_id\": 1, \"start_text\":\"卡房段\",\"to_id\": 2, \"end_text\":\"高松矿床\",\"type\": \"矿化\"}]}\n输出：\n{\"id\": 11, \"text\": \"卡房段就是高松矿床主要的矿化段。\", \"entities\": [{\"id\": 1, \"label\": \"矿段\", \"start_offset\": 0, \"end_offset\": 3,\"text\":\"卡房段\"}, {\"id\": 2, \"label\": \"矿床实例\", \"start_offset\": 5, \"end_offset\": 9,\"text\":\"高松矿床\"}], \"relations\": [{\"id\": 1, \"from_id\": 1,\"start_text\":\"卡房段\", \"to_id\": 2, \"end_text\":\"高松矿床\", \"type\": \"主要\"}, {\"id\": 2, \"from_id\": 1,\"start_text\":\"卡房段\", \"to_id\": 2, \"end_text\":\"高松矿床\", \"type\": \"矿化\"}]}\n{\"id\": 12, \"text\": \"除其他矿段外，卡房段是高松矿床主要的矿化段。\", \"entities\": [{\"id\": 1, \"label\": \"矿段\", \"start_offset\": 8, \"end_offset\": 11,\"text\":\"卡房段\"}, {\"id\": 2, \"label\": \"矿床实例\", \"start_offset\": 12, \"end_offset\": 16,\"text\":\"高松矿床\"}], \"relations\": [{\"id\": 1, \"from_id\": 1, \"start_text\":\"卡房段\",\"to_id\": 2, \"end_text\":\"高松矿床\", \"type\": \"主要\"}, {\"id\": 2, \"from_id\": 1,\"start_text\":\"卡房段\",\"to_id\": 2,\"end_text\":\"高松矿床\", \"type\": \"矿化\"}]}\n{\"id\": 13, \"text\": \"卡房段为高松矿床主要的矿化段。\", \"entities\": [{\"id\": 1, \"label\": \"矿段\", \"start_offset\": 0, \"end_offset\": 3,\"text\":\"卡房段\"}, {\"id\": 2, \"label\": \"矿床实例\", \"start_offset\": 4, \"end_offset\": 8,\"text\":\"高松矿床\"}], \"relations\": [{\"id\": 1, \"from_id\": 1,\"start_text\":\"卡房段\", \"to_id\": 2, \"end_text\":\"高松矿床\", \"type\": \"主要\"}, {\"id\": 2, \"from_id\": 1,\"start_text\":\"卡房段\", \"to_id\": 2, \"end_text\":\"高松矿床\", \"type\": \"矿化\"}]}\n{\"id\": 14, \"text\": \"卡房段是高松矿床最主要的矿化段。\", \"entities\": [{\"id\": 1, \"label\": \"矿段\", \"start_offset\": 0, \"end_offset\": 3,\"text\":\"卡房段\"}, {\"id\": 2, \"label\": \"矿床实例\", \"start_offset\": 4, \"end_offset\": 8,\"text\":\"高松矿床\"}], \"relations\": [{\"id\": 1, \"from_id\": 1,\"start_text\":\"卡房段\",\"to_id\": 2, \"end_text\":\"高松矿床\", \"type\": \"主要\"}, {\"id\": 2, \"from_id\": 1,\"start_text\":\"卡房段\", \"to_id\": 2, \"end_text\":\"高松矿床\", \"type\": \"矿化\"}]}\n{\"id\": 15, \"text\": \"卡房矿段是高松矿田主要的矿化段。\", \"entities\": [{\"id\": 1, \"label\": \"矿段\", \"start_offset\": 0, \"end_offset\": 4,\"text\":\"卡房矿段\"}, {\"id\": 2, \"label\": \"矿床实例\", \"start_offset\": 5, \"end_offset\": 9,\"text\":\"高松矿床\"}], \"relations\": [{\"id\": 1, \"from_id\": 1,\"start_text\":\"卡房矿段\", \"to_id\": 2, \"end_text\":\"高松矿床\", \"type\": \"主要\"}, {\"id\": 2, \"from_id\": 1,\"start_text\":\"卡房矿段\", \"to_id\": 2, \"end_text\":\"高松矿床\", \"type\": \"矿化\"}]}\n\n输入：{\"id\": 25, \"text\": \"总之，成矿主阶段 (I-Ⅲ阶段)发育的包裹体类型多样，均含有沸腾包裹体。\", \"entities\": [{\"id\": 10615, \"label\": \"岩石/矿物微观结构特征\", \"start_offset\": 30, \"end_offset\": 35,\"text\":\"沸腾包裹体\"}, {\"id\": 10697, \"label\": \"成矿阶段、成矿期\", \"start_offset\": 3,\"end_offset\": 16,\"text\":\"成矿主阶段 (I-Ⅲ阶段)\"}], \"relations\": [{\"id\": 10612, \"from_id\": 10697, \"start_text\":\"沸腾包裹体\",\"to_id\": 10615,\"end_text\":\"成矿主阶段 (I-Ⅲ阶段)\",\"type\": \"包含\"}]}\n输出： {\"id\": 2501, \"text\": \"成矿主阶段 (I-Ⅲ阶段)发育的包裹体类型丰富多样，并且均含有沸腾包裹体。\", \"entities\": [{\"id\": 10615, \"label\": \"岩石/矿物微观结构特征\", \"start_offset\": 31, \"end_offset\": 36,\"text\":\"沸腾包裹体\"}, {\"id\": 10697, \"label\": \"成矿阶段、成矿期\", \"start_offset\": 0, \"end_offset\": 13,\"text\":\"成矿主阶段 (I-Ⅲ阶段)\"}], \"relations\": [{\"id\": 10612, \"from_id\": 10697, \"start_text\":\"沸腾包裹体\",\"to_id\": 10615, \"end_text\":\"成矿主阶段 (I-Ⅲ阶段)\",\"type\": \"包含\"}]}, {\"id\": 2502, \"text\": \"发育的成矿主阶段 (I-Ⅲ阶段)中包裹体类型多样，均含有沸腾包裹体。\", \"entities\": [{\"id\": 10615, \"label\": \"岩石/矿物微观结构特征\", \"start_offset\": 28, \"end_offset\": 33,\"text\":\"沸腾包裹体\"}, {\"id\": 10697, \"label\": \"成矿阶段、成矿期\", \"start_offset\": 3, \"end_offset\": 16\",\"text\":\"成矿主阶段 (I-Ⅲ阶段)\"}], \"relations\": [{\"id\": 10612, \"from_id\": 10697, \"start_text\":\"沸腾包裹体\",\"to_id\": 10615,\"end_text\":\"成矿主阶段 (I-Ⅲ阶段)\", \"type\": \"包含\"}]}, {\"id\": 2503, \"text\": \"(I-Ⅲ阶段)成矿主阶段包裹体发育类型多样，同时均含有沸腾包裹体。\", \"entities\": [{\"id\": 10615, \"label\": \"岩石/矿物微观结构特征\", \"start_offset\": 27, \"end_offset\": 32,\"text\":\"沸腾包裹体\"}, {\"id\": 10697, \"label\": \"成矿阶段、成矿期\", \"start_offset\": 0, \"end_offset\": 12,\"text\":\"(I-Ⅲ阶段)成矿主阶段\"}], \"relations\": [{\"id\": 10612, \"from_id\": 10697, \"start_text\":\"沸腾包裹体\",\"to_id\": 10615, \"end_text\":\"(I-Ⅲ阶段)成矿主阶段\",\"type\": \"包含\"}]}, {\"id\": 2504, \"text\": \"(I-Ⅲ阶段)发育的成矿主阶段包裹体丰富多样，并且均含有沸腾包裹体。\", \"entities\": [{\"id\": 10615, \"label\":\"岩石/矿物微观结构特征\", \"start_offset\" :28,\"end_offset\" :33,\"text\":\"沸腾包裹体\" }, {\" id \":10697,\" label \":\" 成矿阶段 、成矿期 \",\"start_offset\" :0 , \" end_offset \":15,\"text\":\"(I-Ⅲ阶段)发育的成矿主阶段\"}], \" relations\":[\" id \":18952,\" from_id \":18762,\"start_text\":\"沸腾包裹体\",\" to_id \":17942,\"end_text\":\"(I-Ⅲ阶段)发育的成矿主阶段\",\" type \":\" 包含 \"] } )\n请在生成过程中严格注意偏移量的变化，请检查增强后entity按偏移量提取内容是否与原本的内容一致，如果不一致，请修改偏移量，再进行输出。"}
                     ] + messages,
            temperature=1,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        print(response.choices[0].message)

        answer = response.choices[0].message.content
        save_answer(str(answer), 'answer.txt')
        time.sleep(1)  # 等待一段时间，避免速率限制



if __name__ == "__main__":
    data = load_data('.\less20.json')
    start_line = 183  # 从第 184 行开始
    process_data(data, start_line)
