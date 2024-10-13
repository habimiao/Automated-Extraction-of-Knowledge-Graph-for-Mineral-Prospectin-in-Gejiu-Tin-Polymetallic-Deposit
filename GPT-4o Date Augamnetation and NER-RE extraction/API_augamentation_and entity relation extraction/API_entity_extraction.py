import time
import json
from openai import OpenAI
api_key = "" #your openai key

client = OpenAI(api_key=api_key)


def load_data(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            try:
                entry = json.loads(line)
                if "text" in entry:
                    texts.append(entry["text"])
            except json.JSONDecodeError:
                continue
    return texts


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
                          "content":  "You will be provided with unstructured data, and your task ：\n请你作为锡矿找矿领域的专家，完成以下任务。\n主要任务：\n请根据语料数据，进行实体部分抽取.\nlet's think it step by step.\n实体抽取：\n实体标签：\n/\n1.矿床知识本体:文献题目，著作章节题目，如“第三章 个旧锡铜多金属矿床深部成矿地质背景”\n2.副标题：一级二级、三级标题，如“3.2 重力致矿异常信息提取”。\n3.矿床：以“矿床”结尾的内容，如“个旧锡铜多金属矿床”、“卡方矿床”、“锡-铜矿床”。\n4.矿床类别：语义阐明矿床类别，如“风化矿床”，“内生矿床”、“热液矿床”、“矽卡岩矿床”。\n5.矿床谱系、矿床系列：以矿床谱系、矿床系列结尾的实体，如“稀有、有色贵金属成矿系列”\n6.区域，矿集区：矿集区，或以“域”“矿集区”结尾的内容如“个旧矿集区”，“变质泥质岩源区”，“同碰撞花岗岩区”\n7.成矿阶段、成矿期：矿床基本知识，如“矽卡岩阶段”，“硫化物阶段”，“高温热液成矿阶段”。\n8.成矿条件特点：成矿特点，成矿条件。根据文中的数据进行判断。\n如“壳-幔混合来源”、“成矿热液多期、多阶段叠加成矿”。\n10.矿体：如“矽卡岩矿体”、“盲矿体”、“深部矿体”等。\n11.矿体部位：矿体的具体部位“矿体中部”，“矿体头部”、“矿体尾部（960m标高）”等。\n12.矿体形态：如“层状矿体”，“脉状矿体”，“浸染状”等。\n13.矿体分布：前文有矿体提示，如“多层分布”，“右倾型”等。\n14.成矿流体：如“含矿流体”，“花岗岩流体”等。\n15.地层：“碳酸岩地层”，“富藻地层”，“硅镁层”等。注意和年代地层区分开。\n16.年代地层：如“三叠系” \"个旧组\"，“T2g1地层”等\n17.地质年代：非具体年代数值，not“2008”等，如“晚白垩世”，“燕山中晚期”，“元古宙”等，。\n18.矿区：如“老厂矿区”，“北炮台矿区”等\n19.矿区位置：如“西区”，“个旧矿田东区”，“老厂矿田”等\n20.矿带、岩石带等：如“花岗岩带”，“外蚀变带”，“哀牢山变质岩带”等\n21.矿段：具体的矿段，如“马拉格段”，“白泥洞段”，“松树脚锡-铅矿段”等，请根据文本判断\n22.矿脉：如“煌斑岩脉”，“电气石-石英-锡石脉”以“脉”结尾的相关的岩石矿物\n23.矿化阶段：前文会矿化相关提示，如“氧化物阶段”，“锡-铜多金属矿化阶段”等，\n24.矿化作用：如“局部矿化作用”，“矽卡岩型硫化物锡-铜矿化”，“矽卡岩化”等\n25.矿化状态：根据前文内容，更加强调状态，如“氧化矿化”，“钾长石化”，“云英岩化”等\n26.蚀变作用：如“羟基蚀变”，“铁染蚀变”，“电气石化”等\n27.岩层：比地层分割粒度细，如“第一层（T2g3-1）”，“第二层（T2g2-2）”，“玄武岩层”等，以“岩层”结尾的文本\n28.矿石结构/构造：根据前文判断，如“浸染状”，“块状构造”，“致密块状”，“皮壳状构造”等\n29.岩石结构：如“斑状中粗粒结构”、“冷凝边”、“斑晶”、“基质”、“接触界面”等\n30.矿物：“毒砂”，“石英”，“萤石”等矿石、矿物名称\n31.矿物形态:根据前文判断，前文有矿物相关提示，如“透镜状”，“等轴状”，“长条状”等\n32.矿物类型:如“副矿物”，“脉石矿物”，“主要矿物”等\n33.岩石类型：根据前文提示，如“贾沙石英二长岩”，“斑状花岗岩”，“等粒花岗岩”等\n34.岩石：如“碳酸盐岩”，“接触交代矽卡岩”，“正长岩”等以“岩”结尾的文本\n35.岩体：如“花岗岩杂岩体”，“东区岩体”，“贾沙斑状石英二长岩体”等\n36.岩体部位：根据前文岩体提示，如“地下深部”，“深部岩体”，“岩体顶部”等\n37.围岩：如“近岩体围岩”，“碳酸盐围岩”，“岩体围岩”等\n38.岩浆：如“同源岩浆”，“花岗质岩浆”，“幔源岩浆”等\n39.岩浆作用：如“壳幔混合岩浆结晶分异作用”，“燕山晚期岩浆活动”，“燕山晚期岩浆活动”等\n40.地质构造/结构：如“哀牢山-红河断裂”，“印支地块”，“小江断裂”，“地幔和/或基底凹陷”等\n41.岩石/矿物微观特征：如“S型子晶”，“流体包裹体”，“VL型包裹体”等\n42.成/容/控矿构造：成矿构造，容矿构造，控矿构造，根据前文判断。\n前有“控制”等字眼，后续接构造，如“褶皱构造”，“断裂构造”，“接触带”等地质构造或结构。\n43.构造形态：如“线性构造”，“环形构造”等，更强调形态。\n44.构造活动：如“个旧断裂活动”，“晚期拉张构造环境”，“张扭活动”等\n45.构造部位：地质构造的具体部位“接触带”，“围岩层理”，“节理裂隙”，“上下盘”等\n46.构造特点：根据前文，强调构造的特点，如“复杂的控岩控矿网络格局”，“张扭性质”，“压扭性”等\n47.地质活动：非“构造活动”下的地质活动，如“壳幔间相互作用”，“岩浆活动”，“陆壳重熔”，“地震活动”。\n48.作用方式：非“矿化作用”，“岩浆作用”，“蚀变作用”，“构造活动”，“地质活动”的情况下，更加具体到作用的形式强调作用的情况下，进行标注，如“断裂变形”，“流体流动”,“侵位”。\n50.作用过程：排除其他标签，本标签更强调过程，注意前后文联系。如“岩浆（热液）演化过程”，“沉积盆地的改造过程”。\n51.作用条件：作用发生的具体条件，为长字符串。如“P>=0.1”,“不同阶段”,\"在加热过程中\"等。\n52.形成原因：强调矿床，矿物等的形成原因，无法以“作用条件”，“作用方式”，“构造活动”，“矿化作用”等标签进行标注，用“地物化条件/特征”等标签标注较为繁琐（多于三个实体三个关系），直接标注为“形成原因”。\n如“某种地质作用使其在局部地段发生了富集或分散”。\n53.地图术语、地质术语：待标实体与关系、成因、特点、条件等有关，同时是地质、地图学相关术语的情况下，无法判断是否属于上面50个实体类型的任意一个，将其标注为“地图术语、地质术语”。对于“图表”的实体在具体段落的名称，也用这个标签进行标注。如“地质体”，“稀土分布模式”，“分异产物”，“中层地质结构”，“岩石结构特征”等\n54.数据：强调原始数据的来源，重力数据，化学异常数据等。如“研究区混合重力数据”，“1：20万重力数据”，“1：20万水系沉积物地球化学数据”等\n55.数据条件、部位:具体描述数据分布情况的位置，数据来源部位。如“风流山34矿区5线剖面”，“谐和线”，“中段”，“下标”，“剔除8个离群数据”等\n56.趋势、数据分布、程度、顺序：趋势（越大，越小，越多等）数据升高，下降，分布（松散、正态分布），程度（强弱）如“较大”，“较宽”，顺序（从大到小）等。如\"由上到下\"，“接近于零”，“大于零”，“频率最低”，“越大”，“越小”等。\n57.属性数值(比例)等：排除“品位”，“储量”，“岩石密度”，“温度范围”，“重力异常范围”之后的数值属性，如“100-120Ma”，“2:1”，“0.17~1.03”，“气相”，“液相”等\n58.化学元素：单个化学元素或者一串元素组合，如“Sb”，“Zn”，“Pb”，“Mo”，“Ag-As-Bi-Cu-F”，“Sn-W-Cu-F-B”等，排除 “成矿元素”，“成晕元素”等模糊词语。\n59.化学组分：如“K2O”，“SiO2”，“K2O”，“NaCl-H₂O”等\n60.序列：如“一级”“二级”等\n61.序列位置：如“序列较前”，“序列前缘”，“序列尾部”等\n62.同位素方法：如“U-Pb同位素年龄”，“辉钼矿Re-Os同位素”，“He同位素”等。\n63.地球化学方法：如“原生晕分带原理”，“微量元素和稀土元素分析”，“同位素地球化学方法”，“地球化学异常探测”等。\n64.地球物理方法：如“磁法勘探”，“重力勘探”，“地震勘探”等\n65.遥感勘探方法：如“雷达遥感（SAR）”，“多光谱和高光谱成像”，“热红外遥感”。\n66.数据统计方法：如“R型因子分析”，“BEMD方法”，“二位经验模分解技术”。\n67.方法、程序、原理：排除上述“地球物理方法”、“地球化学方法”，“遥感勘探方法”，“数据统计方法”后，的方法原理程序，如“地质复杂度模型”，“Isoplot程序”，“ICPMS DataCal程序”等\n68.公式：根据文中引用的公式如“公式7-2”\n69.图表：根据文中标注引用的图表，“图4-6”，“表5-1”，“图5-1~图5-3”等\n70.编号；代码：排除分量，标注样品的具体编号，地图上特定编号等，根据前文，如“F1”，“D4ZK1”等\n71.品位：矿石的品味，一般是数值+%，如“1%”，“0.02%~0.05%”，“3%”等\n72.储量：一般在矿石或矿物后面，和品位一起出现。如“300Mt”，“400Mt”等\n73.岩石密度：如“2.55~2.6g/cm3”,“4.52g/cm3”，“2.98~3.10g/cm3”。\n74.温度范围：如“120~250.2℃”，“100~670℃”等\n75.网距：如“2000m*2000m”\n76.重力异常范围：根据前文判断，如“-3.0~2.5μm/s2”，“-5.5~+2.5μm/s2”等\n77.异常：正异常或负异常，如“负重力异常”，“局部重力异常”。\n78.分量：以分量进行结尾的数据统计术语。如“BIMF1重力分量异常”，“带通滤波重力分量异常BIMF3”，“低通滤波重力分量异常[Res(m,n)]”等\n79.参数：相对较为重要的实验过程中的参数，在文中以地球物理实验设备设置以及地球化学实验各种参数设置等，如“变异性指数（G）”，“[Ba+W]D/[As+Zn+Bi]D”，“GMo”，“Dmax”，“△G”等特指英文符号。\n80.颜色：如“黑白”，“灰色”，“深灰色”，“透明”等\n81.地理位置、方位、方向等：具体的空间地理位置，“云南地区”，方位方向（东西南北），“地壳深部”\n82.地物化/条件：\n  地质物理化学相关条件，温度湿度，倾角，倾向等，矿石产状等，如“倾角”，“冰点”，“均一温度”，“倾向”等。\n  “力学性质”也用这个标签，如“厚度”，“粒径”，“成矿温度”，“总面积”，“含量”，“流体盐度”，“汽液比”等\n83.特点：排除“构造特点”，“成矿特点”后，难以以其他形式进行标注的过程中，反映特点的包含3个实体及关系以上的文本，如“地表未见侵入岩出露”，“顶部平缓、东缓西陡”，“多期多段阶段”。\n84.部门、人员、研究所、实验室：作者认为······、“地质研究所”，“毛景文等”，“河南岩矿分析实验室”等\n85.仪器设备：化学元素分析测试仪器，地磁点震勘探仪器设备。如“双目镜”，“质谱仪”，“Neptune”。\n86.实验处理（样品）：实验处理过程，矿石分选，实验处理。如“挑选锆石”，“结石分选”，“打磨和抛光”,“观察样品的岩相特征”等。\n87.化学参数，术语：对于在文中重要的化学相关术语：如“主要元素含量”，“主要成矿元素”，“稀土元素地球化学特征”，“成矿元素”，“成晕元素”，“分带序列”等在文中对于成矿相对重要的化学相关术语。\n88.数据统计术语：如“均方差”，\"加权平均\"，“顶部标高”，“维氏值”等在文中有具体强调的与成矿元素实验数值相关的数据统计相关术语等。\n/\n请将具体文本中实体标签的文本位置标出并进行抽取成jsonal模式：\n/\n{\"id\":1,\"text\":\"第 3 章...\";\n\"entities\":[{\"id\":1,\"label\":\"矿床实例\",\"start_offset\":45,\"end_offset\":58,\"text\":\"个旧矿床\"},{\"id\":2,\"label\":\"岩石\",\"start_offset\":28,\"end_offset\":33,\"text\":\"花岗岩\"}...]}\n/\nexample：\n\ninput：\n硅酸盐/矽卡岩阶段(I阶段):岩浆期后大量高温含矿气水热液，首先遇到碳酸盐岩类围岩，并在双交代作用和接触渗滤交代作用影响下形成矽卡岩。\noutput：\n{\"id\": 1, \"text\": \"硅酸盐/矽卡岩阶段(I阶段):岩浆期后大量高温含矿气水热液，首先遇到碳酸盐岩类围岩，并在双交代作用和接触渗滤交代作用影响下形成矽卡岩。\", \n\"entities\": [{\"id\": 1, \"label\": \"矿化阶段\", \"start_offset\": 0, \"end_offset\": 14,\"text\":\"硅酸盐/矽卡岩阶段(I阶段)\"}, \n{\"id\": 2, \"label\": \"化学参数，术语\", \"start_offset\": 21, \"end_offset\": 29,\"text\":\"高温含矿气水热液\"}, {\"id\": 3, \"label\": \"围岩\", \"start_offset\": 34, \"end_offset\": 41,\"text\":\"碳酸盐岩类围岩\"}, \n{\"id\": 4, \"label\": \"岩石\", \"start_offset\": 63, \"end_offset\": 66,\"text\":\"矽卡岩\"}, \n{\"id\": 5, \"label\": \"成矿阶段、成矿期\", \"start_offset\": 15, \"end_offset\": 18,\"text\":\"岩浆期\"}, \n{\"id\": 6, \"label\": \"矿化作用\", \"start_offset\": 44, \"end_offset\": 49,\"text\":\"双交代作用\"}, \n{\"id\": 7, \"label\": \"矿化作用\", \"start_offset\": 50, \"end_offset\": 58,\"text\":\"接触渗滤交代作用\"}]}\n\ninput：\n硫化物阶段(Ⅲ阶段):硫化物阶段是个旧矿集区最重要的成矿阶段。\noutput：\n{\"id\": 2, \"text\": \"硫化物阶段(Ⅲ阶段):硫化物阶段是个旧矿集区最重要的成矿阶段。\", \n\"entities\": [{\"id\": 8, \"label\": \"矿化阶段\", \"start_offset\": 0, \"end_offset\": 10,\"text\":\"硫化物阶段(Ⅲ阶段)\"}, \n{\"id\": 9, \"label\": \"区域，矿集区\", \"start_offset\": 17, \"end_offset\": 22,\"text\":\"个旧矿集区\"},\n {\"id\": 10, \"label\": \"成矿阶段、成矿期\", \"start_offset\": 11, \"end_offset\": 16,\"text\":\"硫化物阶段\"}]}]}\n\ninput:\n在深部岩体与围岩接触带上，主要发育矽卡岩型锡-铜矿化。\noutput:\n{\"id\": 3, \"text\": \"在深部岩体与围岩接触带上，主要发育矽卡岩型锡-铜矿化。\", \n\"entities\": [{\"id\": 11, \"label\": \"岩体部位\", \"start_offset\": 1, \"end_offset\": 5,\"text\":\"深部岩体\"}, \n{\"id\": 12, \"label\": \"构造部位\", \"start_offset\": 6, \"end_offset\": 11,\"text\":\"围岩接触带\"}, \n{\"id\": 13, \"label\": \"矿化状态\", \"start_offset\": 14, \"end_offset\": 26,\"text\":\"矽卡岩型锡-铜矿化\"}]]}\n\ninput:\n辉钼矿的晶出标志着硫化物阶段的开始，辉钼矿选择性交代萤石化斜长岩和矽卡岩。\noutput:\n{\"id\": 4, \"text\": \"辉钼矿的晶出标志着硫化物阶段的开始，辉钼矿选择性交代萤石化斜长岩和矽卡岩。\", \n\"entities\": [{\"id\": 14, \"label\": \"矿物\", \"start_offset\": 18, \"end_offset\": 21,\"text\":\"辉钼矿\"}, \n{\"id\": 15, \"label\": \"矿化阶段\", \"start_offset\": 9, \"end_offset\": 17,\"text\":\"硫化物阶段\"},\n {\"id\": 16, \"label\": \"成矿条件\", \"start_offset\": 0, \"end_offset\": 6,\"text\":\"辉钼矿的晶出\"},\n {\"id\": 17, \"label\": \"岩石\", \"start_offset\": 26, \"end_offset\": 32,\"text\":\"萤石化斜长岩\"},\n {\"id\": 18, \"label\": \"岩石\", \"start_offset\": 33, \"end_offset\": 36,\"text\":\"矽卡岩\"}]]}\n\ninput:\n总体上，从西到东，从北到南，花岗岩的形成具有从早到晚，从同碰撞造山期逐渐过渡到碰撞造山期后的演化趋势。\noutput：\n{\"id\": 22, \"text\": \"总体上，从西到东，从北到南，花岗岩的形成具有从早到晚，从同碰撞造山期逐渐过渡到碰撞造山期后的演化趋势。\", \n\"entities\": [{\"id\": 5, \"label\": \"地理位置、方位、方向等\", \"start_offset\": 5, \"end_offset\": 6,\"text\":\"西\"}, \n{\"id\": 2329, \"label\": \"地理位置、方位、方向等\", \"start_offset\": 7, \"end_offset\": 8,\"text\":\"东\"}, \n{\"id\": 2330, \"label\": \"地理位置、方位、方向等\", \"start_offset\": 10, \"end_offset\": 11,\"text\":\"北\"}, \n{\"id\": 2331, \"label\": \"地理位置、方位、方向等\", \"start_offset\": 12, \"end_offset\": 13,\"text\":\"南\"},\n {\"id\": 2332, \"label\": \"成矿特点\", \"start_offset\": 22, \"end_offset\": 50,\"text\":\"从早到晚，从同碰撞造山期逐渐过渡到碰撞造山期后的演化趋势\"},\n {\"id\": 6998, \"label\": \"岩石\", \"start_offset\": 14, \"end_offset\": 17,\"text\":\"花岗岩\"}]}\n\ninput:\n沈渭洲(1987)认为大气降水可以以断裂构造中的自由水或沉积岩中所含的同生水的形式与高温岩浆溶液混合。\noutput:\n{\"id\": 19, \"text\": \"沈渭洲(1987)认为大气降水可以以断裂构造中的自由水或沉积岩中所含的同生水的形式与高温岩浆溶液混合。\", \n\"entities\":\n [{\"id\": 3549, \"label\": \"地质构造/结构\", \"start_offset\": 18, \"end_offset\": 22,\"text\":\"断裂构造\"},\n {\"id\": 3550, \"label\": \"岩石\", \"start_offset\": 28, \"end_offset\": 31,\"text\":\"沉积岩\"}, \n{\"id\": 3551, \"label\": \"岩浆\", \"start_offset\": 42, \"end_offset\": 48,\"text\":\"高温岩浆\"}, \n{\"id\": 9835, \"label\": \"部门、人员、研究所、实验室\", \"start_offset\": 0, \"end_offset\": 3,\"text\":\"沈渭洲\"}, \n{\"id\": 9837, \"label\": \"地图术语、地质术语\", \"start_offset\": 11, \"end_offset\": 15,\"text\":\"大气降水\"}, {\"id\": 9838, \"label\": \"化学参数，术语\", \"start_offset\": 24, \"end_offset\": 27,\"text\":\"自由水\"}, \n{\"id\": 9839, \"label\": \"化学参数，术语\", \"start_offset\": 35, \"end_offset\": 38,\"text\":\"同生水\"}]}\n\ninput:\n其余9个点中，3个点年龄偏大(87～88Ma),6个点年龄偏小(68.2～76Ma), 故不参与年龄加权平均计算。\noutput:\n{\"id\": 24, \"text\": \"其余9个点中，3个点年龄偏大(87～88Ma),6个点年龄偏小(68.2～76Ma), 故不参与年龄加权平均计算。\", \"entities\": [{\"id\": 7173, \"label\": \"属性：数值(比例)等\", \"start_offset\": 15, \"end_offset\": 22,\"text\":\"87～88Ma\"}, {\"id\": 7176, \"label\": \"属性：数值(比例)等\", \"start_offset\": 7, \"end_offset\": 10,\"text\":\"3个点\"}, {\"id\": 7177, \"label\": \"属性：数值(比例)等\", \"start_offset\": 24, \"end_offset\": 27,\"text\":\"6个点\"}, {\"id\": 7178, \"label\": \"属性：数值(比例)等\", \"start_offset\": 32, \"end_offset\": 41,\"text\":\"68.2～76Ma\"}, {\"id\": 9348, \"label\": \"属性：数值(比例)等\", \"start_offset\": 2, \"end_offset\": 5,\"text\":\"9个点\"}, {\"id\": 9349, \"label\": \"趋势、数据分布、程度、顺序\", \"start_offset\": 12, \"end_offset\": 14,\"text\":\"偏大\"}, {\"id\": 9350, \"label\": \"趋势、数据分布、程度、顺序\", \"start_offset\": 29, \"end_offset\": 31,\"text\":\"偏小\"}]}\n\ninput\n总之，北炮台斑状 黑云母花岗岩老于贾沙石英二长岩，而老卡花岗岩具有较宽的年龄分布。\n\noutput\n{\"id\": 24, \"text\": \"总之，北炮台斑状 黑云母花岗岩老于贾沙石英二长岩，而老卡花岗岩具有较宽的年龄分布。\",\n \"entities\": \n[{\"id\": 2088, \"label\": \"岩石\", \"start_offset\": 3, \"end_offset\": 15,\"text\":\"北炮台斑状 黑云母花岗岩\"},\n {\"id\": 2090, \"label\": \"岩石\", \"start_offset\": 17, \"end_offset\": 24,\"text\":\"贾沙石英二长岩\"},\n {\"id\": 2092, \"label\": \"岩石\", \"start_offset\": 26, \"end_offset\": 31,\"text\":\"老卡花岗岩\"},\n {\"id\": 7773, \"label\": \"趋势、数据分布、程度、顺序\", \"start_offset\": 33, \"end_offset\": 35,\"text\":\"较宽\"}, \n{\"id\": 7774, \"label\": \"数据统计术语\", \"start_offset\": 36, \"end_offset\": 40,\"text\":\"年龄分布\"}]}\n\ninput\n矽卡岩阶段和氧化物阶段包裹体发育较小且较少，但含子晶包裹体 (S型包裹体)较多；硫化物阶段流体包裹体最为发育，数量多且较大，该阶段还发育C型包裹体。\noutput：\n{\"id\": 25, \"text\": \"矽卡岩阶段和氧化物阶段包裹体发育较小且较少，但含子晶包裹体 (S型包裹体)较多；硫化物阶段流体句裹体最为发育，数量多且较大，该阶段还发育C型包裹体。\", \n\"entities\": [\n{\"id\": 10611, \"label\": \"岩石/矿物微观结构特征\", \"start_offset\": 68, \"end_offset\": 73,\"text\":\"C型包裹体\"},\n {\"id\": 10612, \"label\": \"岩石/矿物微观结构特征\", \"start_offset\": 31, \"end_offset\": 36,\"text\":\"S型包裹体\"},\n {\"id\": 10698, \"label\": \"成矿阶段、成矿期\", \"start_offset\": 0, \"end_offset\": 5,\"text\":\"矽卡岩阶段\"},\n {\"id\": 10699, \"label\": \"成矿阶段、成矿期\", \"start_offset\": 6, \"end_offset\": 14,\"text\":\"氧化物阶段\"},\n {\"id\": 10700, \"label\": \"成矿阶段、成矿期\", \"start_offset\": 40, \"end_offset\": 45,\"text\":\"硫化物阶段\"},\n {\"id\": 10704, \"label\": \"特点\", \"start_offset\": 50, \"end_offset\": 61,\"text\":\"最为发育，数量多且较大\"}]}\n注意entity中的包含\"id\" \"label\" \"start_offset\" \"end_offset\" \"text\"，请以代码生成的格式进行处理，保证label为上述我提供的标签，\"start_offset\" \"end_offset\"为偏移量，根据偏移量计算的\"text\"，能一一对应，请尽可能提取实体标签。\n\ninput：\n首先计算各成矿成晕元素在每个中段的分带指数(表9-3),初步得出原生晕的分带序列为：[F-B-Ba-W]→ [Cu-Sb]→[Be-Cr]→[Sn-Pb-Zn-Ni-Ag]→[Co-Mo-Bi-As]。\noutput\n{\"id\": 17, \"text\": \"首先计算各成矿成晕元素在每个中段的分带指数(表9-3),初步得出原生晕的分带序列为：[F-B-Ba-W]→ [Cu-Sb]→[Be-Cr]→[Sn-Pb-Zn-Ni-Ag]→[Co-Mo-Bi-As]。\", \n\"entities\": [\n{\"id\": 5559, \"label\": \"图表\", \"start_offset\": 22, \"end_offset\": 26,\"text\":\"表9-3\"},\n{\"id\": 8988, \"label\": \"分带序列\", \"start_offset\": 42, \"end_offset\": 52,\"text\":\"[F-B-Ba-W]\"}, \n{\"id\": 8989, \"label\": \"分带序列\", \"start_offset\": 54, \"end_offset\": 61,\"text\":\"[Cu-Sb]\"}, \n{\"id\": 8990, \"label\": \"分带序列\", \"start_offset\": 62, \"end_offset\": 69,\"text\":\"[Be-Cr]\"}, \n{\"id\": 8991, \"label\": \"分带序列\", \"start_offset\": 70, \"end_offset\": 86,\"text\":\"[Sn-Pb-Zn-Ni-Ag]\"},\n {\"id\": 8992, \"label\": \"分带序列\", \"start_offset\": 87, \"end_offset\": 100,\"text\":\"[Co-Mo-Bi-As]\"},\n {\"id\": 8993, \"label\": \"地图术语、地质术语\", \"start_offset\": 32, \"end_offset\": 40,\"text\":\"原生晕的分带序列\"}\n]}\ninput：\n富液相包裹体(LV型)指室温下主要由液相和一个小气泡组成的二相、加热均一至液相的包裹体，这类包裹体发育最为广泛，在各矿床各阶段均有发育，大小3～86μm，气/液比5%～50%(10%～30%居多)，与富气相共存时，可见气/液比为55%的富液相包裹体。\noutput：\n{\"id\": 25, \"text\": \"富液相包裹体(LV型)指室温下主要由液相和一个小气泡组成的二相、加热均一至液相的包裹体，这类包裹体发育最为广泛，在各矿床各阶段均有发育，大小3～86μm，气/液比5%～50%(10%～30%居多)，与富气相共存时，可见气/液比为55%的富液相包裹体。\", \"entities\": [\n{\"id\": 10606, \"label\": \"岩石/矿物微观结构特征\", \"start_offset\": 0, \"end_offset\": 11,\"text\":\"富液相包裹体(LV型)\"}, \n{\"id\": 10616, \"label\": \"岩石/矿物微观结构特征\", \"start_offset\": 118, \"end_offset\": 124,\"text\":\"富液相包裹体\"}, \n{\"id\": 10677, \"label\": \"属性：数值(比例)等\", \"start_offset\": 70, \"end_offset\": 76,\"text\":\"3～86μm\"}, \n{\"id\": 10678, \"label\": \"属性：数值(比例)等\", \"start_offset\": 81, \"end_offset\": 87,\"text\":\"5%～50%\"}, \n{\"id\": 10679, \"label\": \"属性：数值(比例)等\", \"start_offset\": 88, \"end_offset\": 95,\"text\":\"10%～30%\"}, \n{\"id\": 10680, \"label\": \"属性：数值(比例)等\", \"start_offset\": 114, \"end_offset\": 117,\"text\":\"55%\"}, \n{\"id\": 10687, \"label\": \"特点\", \"start_offset\": 12, \"end_offset\": 67,\"text\":\"室温下主要由液相和一个小气泡组成的二相、加热均一至液相的包裹体，这类包裹体发育最为广泛，在各矿床各阶段均有发育\"}, \n{\"id\": 10688, \"label\": \"地物化条件/特征\", \"start_offset\": 77, \"end_offset\": 81,\"text\":\"气/液比\"}, \n{\"id\": 10689, \"label\": \"地物化条件/特征\", \"start_offset\": 109, \"end_offset\": 113,\"text\":\"气/液比\"}, \n{\"id\": 10690, \"label\": \"作用条件\", \"start_offset\": 99, \"end_offset\": 106,\"text\":\"与富气相共存时\"}\n]}\ninput:\n测温前将样品磨制成约0.3mm厚的包裹体测温片，然后在显微镜下观察样品的岩相特征，结合野外调查和光薄片鉴定确定结果，观察主矿物和包裹体特征。\noutput:\n{\"id\": 25, \"text\": \"测温前将样品磨制成约0.3mm厚的包裹体测温片，然后在显微镜下观察样品的岩相特征，结合野外调查和光薄片鉴定确定结果，观察主矿物和包裹体特征。\", \n\"entities\": [\n{\"id\": 10578, \"label\": \"实验处理（样品）\", \"start_offset\": 6, \"end_offset\": 23 ,\"text\": \"磨制成约0.3mm厚的包裹体测温片\"}, \n{\"id\": 10579, \"label\": \"仪器设备\", \"start_offset\": 27, \"end_offset\": 30, \"text\": \"显微镜\"}, \n{\"id\": 10580, \"label\": \"实验处理（样品）\", \"start_offset\": 31, \"end_offset\": 40, \"text\": \"观察样品的岩相特征\"}, \n{\"id\": 10581, \"label\": \"实验处理（样品）\", \"start_offset\": 58, \"end_offset\": 69, \"text\": \"观察主矿物和包裹体特征\"}]\n}\ninput:\n其结果表明，局部重力异常的变化可能主要受花岗杂岩体和上覆碳酸盐岩的厚度，以及局部矿化作用的形态变化影响。\noutput:\n{\"id\": 13, \"text\": \"其结果表明，局部重力异常的变化可能主要受花岗杂岩体和上覆碳酸盐岩的厚度，以及局部矿化作用的形态变化影响。\", \n\"entities\": [\n{\"id\": 709, \"label\": \"岩体\", \"start_offset\": 20, \"end_offset\": 25,\"text\":\"花岗杂岩体\"},\n{\"id\": 713, \"label\": \"异常\", \"start_offset\": 6, \"end_offset\": 12,\"text\":\"局部重力异常\"}, \n{\"id\": 6394, \"label\": \"岩体\", \"start_offset\": 26, \"end_offset\": 32,\"text\":\"上覆碳酸盐岩\"}, \n{\"id\": 6397, \"label\": \"地物化条件/特征\", \"start_offset\": 33, \"end_offset\": 35,\"text\":\"厚度\"}, \n{\"id\": 6398, \"label\": \"矿化作用\", \"start_offset\": 38, \"end_offset\": 44,\"text\":\"局部矿化作用\"}, \n{\"id\": 6399, \"label\": \"地图术语、地质术语\", \"start_offset\": 45, \"end_offset\": 49,\"text\":\"形态变化\"}\n]\n}\n注意：\n  “较少”,“较宽”等表示数量程度趋势的也要以“趋势/数据分布/顺序”实体标签，进行标注。\n注意“化学元素”不标注模糊的元素术语如“成矿元素”，“成云元素”等这些模糊的元素术语标注为“化学参数，术语”\n注意“矿段”不标注“上段”，“下段”等，这些标注为“数据条件、部位”。\n注意文中“压力公式”，“饱和压力公式”等非具体“编号”+“公式”标注为“数据统计术语”\n  文本中的数值也要进行抽取如百分比（35%），具体数值如年龄（87～88Ma）等，如若不清楚具体数值标签，用\"属性：数值(比例)等\"标签进行抽取,\n输出的实体格式为一行一个实体，如下：\n\"entities\": [\n      {\"id\": 1,\"label\": \"岩石\",\"start_offset\": 0,\"end_offset\": 6,\"text\": \"斑状花岗岩\"},\n      {\"id\": 2,\"label\": \"岩石\",\"start_offset\": 7,\"end_offset\": 18,\"text\": \"北炮台斑状黑云母花岗岩\"},\n      ....\n]",
          }] + messages,
            temperature=0,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        print(response.choices[0].message)

        answer = response.choices[0].message.content
        save_answer(str(answer), 'NER.txt')
        time.sleep(1)  # 等待一段时间，避免速率限制



if __name__ == "__main__":
    data = load_data('.\data_origin.json')
    start_line = 0  # 从第 184 行开始
    process_data(data, start_line)
