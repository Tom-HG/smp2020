# -*- coding: utf-8 -*-
import os
import re
import emoji
import json
from langconv import Converter


def process(data, labels, drop_empty=False):
    x = []
    y = []
    for content, label in zip(data, labels):
        content = re.sub(r'(//@.*?:)', '', content)  # 微博用户昵称，转发标记
        content = re.sub(r'http://[0-9a-zA-Z\./]+', '', content)  # 超链接
        content = re.sub(r'回复@.*?:', '', content)  # 回复某人
        # content = re.sub(r'(@TFBOYS-王俊凯)|(@新闻)|(@北京旅游)|(@兰州公安)|(@兰州发布)|(@中国电信)|'
        #                  r'(@浙江电信)|(@X玖少年团肖战DAYTOY)|(@E网邮情)|(@青岛市)|(@服务)|(@央视新闻)|(@深圳交警)|'
        #                  r'(@Chenmaoyun123)|(@聚美优品)|(@聚美时尚馆)|(@聚美优品客服中心)|(@中国政府网)|'
        #                  r'(@部落冲突)|(@罗志祥)|(@胡海泉)|(@至上励合)|(@交通银行信用卡中心)|(@北京地铁)|'
        #                  r'(@金国秋)|(@北京交通广播)|(@哲思图书)|(@哲思杂志)|(@那只虎斑猫)|(@平顶山市交通局)|(@建行)|'
        #                  r'(@演员马可)|(@微博雷达)|(@绵羊卷)|(@北京人不知道的北京事儿)|(@看见--2009)|(@新浪小小秘书)|(@唐艺昕)|'
        #                  r'(@毛晓彤)|(@温江人不知道的温江事)|(@我的少女时代)|(@HiRumoom)|(@姚志宏DD)|(@Les_etoiles-肖战公益站)|'
        #                  r'(@镇江人)|(@澎湃新闻)|(@米斯特旦欧欧)|(@江苏广播)|(@蔡徐坤)|(@努力努力再努力)|(@小辫儿张云雷)|'
        #                  r'(@郑爽SZ)|(@UNIQ-王一博)|(@雷军)|(@ZJFF社团联合会)|(@王晨MuKii)|(@花开尐陌)|(@Noodler毛毛)|'
        #                  r'(@虎门太平)|(@广州地铁)|(@苏荷半道)|(@招商银行)|(@招商银行信用卡)|(@青岛交通广播FM)|(@魅族科技)|'
        #                  r'(@白永祥)|(@黄章)|(@NBA)|(@自来水公司)|(@温骆Eo)|(@宁波大小事)|(@肖巨牌)|(@宁波发布)|(@宁波市公安局交通警察局)|'
        #                  r'(@商广网)|(@商丘网)|(@微博商丘)|(@有道云笔记)|(@顺丰速运官微)|(@顺丰官方客服)|(@邓超)|(@微博等级)|'
        #                  r'(@新浪微博)|(@天津身边事)|(@天津生活通)|(@新浪天津)|(@肯德基)|(@长春饭店小奶油)|(@长春头条资讯)|(@木瓜芒果)|'
        #                  r'(@多米音乐)|(@上海交警)|(@上海铁路公安处虹桥站派出所)|(@我不说话的时候都是在害羞)|(@小米公司)|(@中国农业银行磐东支行)|'
        #                  r'(@滴滴打车)|(@中国联通)|(@李易峰)|(@四川移动)|(@中国移动)|(@人民日报)|(@M鹿M)|'
        #                  r'(@泡芙会膨胀)|(@麦美娟Alice香港立法会议)|(@Smile-咿呀咿呀呦)|(@湖北之声)|(@王广允star)|(@华晨宇yu)|'
        #                  r'(@Dear迪丽热巴)|(@宝鸡交通)|(@平安拉萨)|(@皎若新月)|(@共青团)|(@李鑫一Rex)|(@腾讯新闻)|(@朱一龙)|'
        #                  r'(@好运一只龙)|(@三联生活周刊)|(@CCTV-10科教频道)|(@环球时报)|(@广州新闻)|(@默贱贱)|(@山下小兰芽)|(@橘子-Fan)|'
        #                  r'(@firstkiss)|(@Cherry_张z)', '', content)  # @某人，非标准格式，单独删除
        content = re.sub(r'[【】/“”\s·。➕Этойвлшебню\'архчьЯмгсу■\-=●♀♂\?]', '', content)  # 无意义的符号
        content = re.sub(r'(\(͡°͜ʖ͡°\))|(o\(>﹏\)o)|(╭\(╯\^╰\)╮)|(\^0\^)|(\(ω\)★)|(→_→)|(•̭̆•̆)|'
                         r'(（｡ò∀ó｡）)|(_\(:_」∠\)_)|(↖\(ω\)↗)|(╮⊙o⊙╭╮)|(\(╥ω╥`\))|'
                         r'(&amp;gt;)|(&amp;quot;)|(\-_\-\|\|)|(\(◍˃̶ᗜ˂̶◍\)ﾉ)|(\(ง•̀_•́\)ง)|(\(•̀ω•́\)✧)|'
                         r'(\(╯‵□′\)╯︵┻━┻)|(\(#ﾟДﾟ\)！)|(~\(>_<\)~)|(\(๑•̀ω•́๑\))|(\(´･ω･`\))|'
                         r'(ヾ\(Ő∀Ő๑\)ﾉ)|(~\\\(≧▽≦\)/~)|(\(づ￣3￣\)づ)|(\(●✿∀✿●\))|(\(￣▽￣\))|'
                         r'(\(\(\(￣へ￣井\))|(\(ಥ_ಥ)\)', '', content)  # 颜文字
        content = re.sub(r'[0-9\.%]+', 'NUM', content)  # 纯数字
        content = Converter("zh-hans").convert(content)  # 繁体字转简体字
        if len(content) == 0:
            if not drop_empty:
                x.append('空')
                y.append(label)
        else:
            x.append(content)
            y.append(label)
    return x, y


def process_emoji(data, emoji_dict):
    """
    将图片表情转换成文字表情，不在字典中的删除
    ❤ -> [心]
    """
    # 转换字典
    emoji2zh = emoji_dict

    result = []
    for content in data:
        if emoji.emoji_count(content) > 0:
            for e in emoji.emoji_lis(content):
                if e['emoji'] in emoji2zh:
                    replace = '[{}]'.format(emoji2zh[e['emoji']])
                    content = content.replace(e['emoji'], replace)
                else:
                    # 出现未收录的表情，删除
                    content = content.replace(e['emoji'], '')
            if len(content) == 0:
                result.append('空')
            else:
                result.append(content)
        else:
            # 无图片表情，原样输出
            result.append(content)

    return result


if __name__ == '__main__':
    # paths = [
    #     r'../raw/train/usual_train.txt',
    #     r'../raw/train/virus_train.txt',
    #     r'../raw/eval/usual_eval.txt',
    #     r'../raw/eval/virus_eval.txt'
    # ]

    # 分开统计
    # for p in paths:
    #     print(os.path.basename(p))
    #     with open(p, 'r', encoding='utf-8') as f:
    #         temp = json.load(f)
    #     emoji_count([i['content'] for i in temp])

    # 合并统计
    # data = []
    # for p in paths:
    #     print(os.path.basename(p))
    #     with open(p, 'r', encoding='utf-8') as f:
    #         temp = json.load(f)
    #     data.extend([i['content'] for i in temp])
    #
    # data = process_emoji(data)
    # for d in data:
    #     print(d)
    pass
