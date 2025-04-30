import re

import torch
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from .models import BitStringMapper, ContrastiveTrainer, module_load_state_dict
from sklearn.metrics.pairwise import cosine_similarity

sf = [
    "“本源悟空”运行十亿级AI微调大模型",
    "近日，本源量子、合肥综合性国家科学中心人工智能研究院等机构联合攻关，在第三代自主超导量子计算机“本源悟空”真机上，实现了全球首次运行十亿参数AI大模型微调任务",
    "空间计算技术让网购“身临其境”",
    "日前，著名美妆品牌罗拉·玛斯亚宣布与体验式电商平台Obsess合作，共同推出面向XR（扩展现实）端的Vision版美妆购物应用",
    "这一应用允许用户按类别购物，并且以“身临其境”的方式虚拟访问位于纽约和巴黎的精品店",
    "这是“对等关税”，还是“滥施普课”关税？",
    "具有讽刺意味的是，英文Reciprocal一词，除了常见的“互惠的、对等的”含义以外，还有“反向的、互逆的，互反的”的意思",
    "美方所谓“对等关税”，实则滥用霸权主义、挥舞关税大棒，有专家将其音译为“滥施普课”关税",
    "1930年，美国出台的《斯姆特-霍利关税法》，曾试图通过加征关税保护本土产业，但其引发的关税战，最终加剧了大萧条",
    "音犹在耳，前事勿忘",
    "放下这份逆全球化、逆全球人心的“滥施普课”关税，与贸易伙伴通过平等对话妥善解决分歧，方为正道",
    "数读春日经济洞见市场活力",
    "万物生长的春天里，文旅市场也呈现出勃勃生机",
    "从踏青赏花到寻味尝鲜，从徒步骑行到爬山露营，都不乏亮眼的市场数据和消费趋势，彰显出春日经济的供需两旺、活力十足",
    "春日经济叠加假日经济，消费场景、各类服务和产品不断上新，为美好生活增添了更多色彩，也为经济社会发展注入了勃勃生机",
    "国内油价年内第三涨 国际油价短期宽幅波动运行",
    "国家发改委价格监测中心监测，本轮成品油调价周期内（3月19日至4月1日），国际油价震荡上行",
    "国家发改委价格监测中心预计，长期来看，国际油价仍存下行压力；短期内，受国际贸易环境变化、地缘政治局势动荡等因素叠加影响，国际油价将呈现宽幅波动走势",
    "至此，国内油价迎来年内第三次上调",
    "据测算，以油箱容量50升的普通私家车计算，车主加满一箱92号汽油多花9元左右",
    "今年以来，国内油价已经历7轮调整，呈现“三涨三跌一搁浅”格局",
    "除海南和西藏外，其它地区92号汽油价格继续维持在“7元”时代",
    "从“甲亢哥”直播看中医药文化“破圈”",
    "近期，美国网红主播“甲亢哥”的中国行在海内外社交媒体上获得关注",
    "这名00后博主在四川成都到同仁堂诊疗的经历，成为中医药文化对外传播的生动注脚",
    "讲好中医药故事，我们要以开放心态拥抱现代话语体系，让传统智慧与当代科学对话，让中医药作为守护人类健康的一剂“良方”更好发挥作用",
    "人工林面积世界第一，为啥还要年年植树？",
    "森林是陆地生态的主体，人类生存的根基",
    "除了净化空气、保持水土、防风固沙，森林还集水库、粮库、钱库、碳库于一身，是不折不扣的大宝库",
    "绿水青山就是金山银山",
    "依托森林宝库，咱们的木竹加工、森林食品、林下经济、生态旅游4个支柱产业的年产值都已超万亿元",
    "还有林草年碳汇量也超过12亿吨二氧化碳当量，居世界首位",
    "生态美、产业兴、百姓富，未来还得把这座金山银山做得更大",
    "凭一手点心绝活拿到人才认定",
    "左手半握裹着馅的面皮，右手一捏一提一旋，不过几秒钟，一个有着18道褶子“鲫鱼嘴”的杭州小笼包现于掌心",
    "用餐高峰时段，浙江省杭州市西湖边的知味观湖滨总店面点间里，面点师李仪的一套动作如行云流水",
    "青春同行 续写友谊新篇章",
    "国之交在于民相亲，民相亲要从青年做起",
    "更开放、更自信、更充满活力的中国，正吸引着越来越多的外国青年",
    "他们走访不同城市，观察中国发展，增进对中国的了解，续写着青年一代新的友好故事",
    "维护多边贸易体制才能实现共同繁荣",
    "经济全球化时代，任何以邻为壑的短视政策都将损人害己",
    "护多边贸易体制是保障各方利益、实现共同繁荣的必由之路",
    "我们敦促美方摒弃零和思维，立即取消单边关税措施，回到对话合作的正确轨道上来，与贸易伙伴平等协商，妥善解决分歧，造福各国人民",
    "办公室保健操丨春季提升免疫力 试试九步搓耳操",
    "从经络学角度来说，人体很多经络都与耳朵有联系",
    "广东省中医院老年医学科主任中医师李际强在接受人民网采访时介绍，通过搓揉耳朵，加速耳部的血液循环，刺激耳部相应穴位，能调节脏腑功能及全身气血运行，起到提高人体免疫力，保健养生的作用",
    "小心果糖的“甜蜜陷阱”",
    "数据显示，我国成年人高尿酸血症患病率为14%，且呈年轻化趋势",
    "在全民关注健康、防控“三高”的今天，这个被称为“第四高”的代谢性疾病正引起公众的广泛关注",
    "众所周知，高尿酸血症与痛风的发生、膳食及生活方式等密切相关",
    "喝啤酒、吃海鲜会让尿酸升高，这已经不是什么新鲜事，但你知道吗？",
    "高果糖食物也是导致尿酸升高的重要因素",
    "越来越多研究证实，包括鲜榨果汁等在内的高果糖饮食是引起高尿酸血症的原因之一，并与代谢综合征密切相关",
    "护佑女性健康 中医这样开方",
    "近年来，国家中医药管理局会同有关部门积极促进中医药与妇幼健康融合发展，加强中医医院妇科建设，推动妇幼保健院设置中医临床科室，充分发挥中医药在服务保障女性健康中的优势作用",
    "人民热评：学生餐的每一分钱都不容“黑吞”",
    "任何时候，都不容贪腐的黑手、油手伸向学生餐桌",
    "当前，全国中小学校园食品安全和膳食经费管理突出问题专项整治正在向纵深持续推进",
    "教育部也表示，今年将狠抓制度落实，紧盯食材采购、配送、查验、贮存、加工制作等关键环节，严格执行经费管理制度，健全安全风险防控体系",
    "校园餐必须是阳光餐、良心餐、放心餐",
    "严查违纪违法、严防漏洞空隙、严管人员环节，才能共同守护好那一张饭桌，不辜负那一份期待",
    "眼睛亮、见事早、行动快（人民论坛）",
    "做到眼睛亮、见事早、行动快，一靠能力，二靠责任",
    "不论是防范化解各类风险隐患，还是纵深推进全面从严治党，抑或是抢抓转瞬即逝的发展机遇，都要求我们提高敏锐度、下好先手棋、打好主动仗",
    "什么是领导干部的能力？",
    "古人讲“见微以知萌，见端以知末，故见象箸而怖，知天下不足也”",
    "善于看到地平线以下的事物，并在桅杆顶刚刚露出的时候就掌握住它，这要求我们必须提高工作的预见性、前瞻性",
    "人民热评：不能容忍“开盒挂人”污染网络空间",
    "“开盒挂人”有某种顽固性，不时见诸舆论",
    "首先是因为“开盒挂人”具有较强的技术属性，往往有一定的隐蔽性",
    "像开头提到的案件中，一人是网络信息工程专业的博士，主攻就是网络安全",
    "较高的技术门槛，事实上形成了一层屏障，增加了防范侦破的难度",
    "另一方面，在于“开盒挂人”形成了境内外违法组织的合流",
    "比如不少案件中都出现了境外社工库的身影，大量的公民个人信息被泄露之后，会被传到境外的社工库上",
    "由于信息在境外，这就给执法造成了障碍",
    "贸易战没有赢家，保护主义不是出路",
    "世界各国经济相互依存、相互联系",
    "中国需要美国，美国也需要中国",
    "双方应加强沟通，畅通对话渠道，遵循市场规律，共同维护公平竞争的国际经贸秩序，为世界经济注入亟需的确定性"
]


def process_text_watermark(text_file):
    # 加载预训练模型
    model = ContrastiveTrainer()
    avg_watermark = []
    try:
        checkpoint = torch.load('staticfiles/watermarking_10_best.pth', map_location='cpu',
                                weights_only=False)
        module_load_state_dict(model, checkpoint['model_state_dict'])
    except Exception as e:
        print(f"加载模型错误: {e}")

    all_text_embeddings = []

    # 处理每个文本
    for text in text_file:
        # 将文本转换为模型输入格式
        inputs = text

        # 生成水印
        with torch.no_grad():
            try:
                text_emb = model(inputs)
                all_text_embeddings.append(text_emb.detach().numpy())
                avg_text_embeddings = torch.tensor(np.mean(all_text_embeddings, axis=0))
            except Exception as e:
                print(f"处理文本 '{text}' 时出现错误: {e}")
                continue

        # 保存水印和文本嵌入
        avg_watermark = BitStringMapper().forward(avg_text_embeddings)
    # 计算平均水印和平均文本嵌入

    return avg_watermark


def calculate_similarity(watermark1, watermark2):
    if watermark1 is None or watermark2 is None:
        return None
    # 确保输入是二维数组
    watermark1 = np.array(watermark1).reshape(1, -1)
    watermark2 = np.array(watermark2).reshape(1, -1)

    # 计算余弦相似度
    similarity = cosine_similarity(watermark1, watermark2)[0][0]

    return similarity


def get_posbility(s_f, t_f):
    results = process_text_watermark(s_f).detach().cpu().numpy()
    results1 = process_text_watermark(t_f).detach().cpu().numpy()
    re1 = calculate_similarity(results, results1)
    return re1


def index(request):
    """
    显示首页
    """
    return render(request, 'index.html')


def process_news_text(request):
    """
    处理从index.html提交的新闻文本或文件

    Args:
        request: HTTP请求对象，包含POST数据或文件

    Returns:
        JsonResponse: 包含处理结果的JSON响应
    """
    if request.method == 'POST':
        try:
            # 获取文本内容（可能是直接输入的文本或文件内容）
            news_text = request.POST.get('text', '')
            uploaded_file = request.FILES.get('file')
            print(news_text)
            if not news_text and not uploaded_file:
                return JsonResponse({
                    'status': 'error',
                    'message': '请输入新闻内容或上传文件'
                })

            # 如果上传了文件，读取文件内容
            if uploaded_file:
                try:
                    # 假设文件是文本文件
                    news_text = uploaded_file.read().decode('utf-8')
                except Exception as e:
                    return JsonResponse({
                        'status': 'error',
                        'message': f'文件读取错误: {str(e)}'
                    })

            # 将文本按换行符分割，第一行通常是标题
            lines = news_text.split('\n')
            title = lines[0].strip() if lines else ''
            content = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ''

            # 将内容分割成句子
            sentences = re.split(r'[。！？]', content)
            # 过滤掉空字符串和只包含空格的字符串
            sentences = [s.strip() for s in sentences if s.strip()]

            # 如果标题不为空，将其添加到句子列表的开头
            if title:
                sentences.insert(0, title)

            try:
                # 获取参数
                # args = parse_args()

                # 计算水印相似度
                similarity = get_posbility(sf, sentences)

                if similarity is None:
                    return JsonResponse({
                        'status': 'error',
                        'message': '无法计算相似度'
                    })

                # 将相似度转换为百分比
                probability = "{:.2f}".format(float(round(similarity * 100, 1)))

                # 返回结果
                return JsonResponse({
                    'status': 'success',
                    'probability': probability,
                    'message': '处理成功'
                })
            except Exception as e:
                print(f"水印计算错误: {str(e)}")  # 添加调试信息
                return JsonResponse({
                    'status': 'error',
                    'message': f'水印计算错误: {str(e)}'
                })

        except Exception as e:
            print(f"处理出错: {str(e)}")  # 添加调试信息
            return JsonResponse({
                'status': 'error',
                'message': f'处理出错: {str(e)}'
            })

    return JsonResponse({
        'status': 'error',
        'message': '请使用POST方法提交数据'
    })
