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
        content = re.sub(r'[【】/“”\s·➕ЭтойвлшебнюархчьЯмгсу■=●♀♂\?]', '', content)  # 无意义的符号
        content = re.sub(r'(\(͡°͜ʖ͡°\))|(o\(>﹏\)o)|(╭\(╯\^╰\)╮)|(\^0\^)|(\(ω\)★)|(→_→)|(•̭̆•̆)|'
                         r'(（｡ò∀ó｡）)|(_\(:_」∠\)_)|(↖\(ω\)↗)|(╮⊙o⊙╭╮)|(\(╥ω╥`\))|'
                         r'(&amp;gt;)|(&amp;quot;)|(\-_\-\|\|)|(\(◍˃̶ᗜ˂̶◍\)ﾉ)|(\(ง•̀_•́\)ง)|(\(•̀ω•́\)✧)|'
                         r'(\(╯‵□′\)╯︵┻━┻)|(\(#ﾟДﾟ\)！)|(~\(>_<\)~)|(\(๑•̀ω•́๑\))|(\(´･ω･`\))|'
                         r'(ヾ\(Ő∀Ő๑\)ﾉ)|(~\\\(≧▽≦\)/~)|(\(づ￣3￣\)づ)|(\(●✿∀✿●\))|(\(￣▽￣\))|'
                         r'(\(\(\(￣へ￣井\))|(\(ಥ_ಥ\))', '', content)  # 颜文字
        # content = re.sub(r'[0-9\.%]+', 'NUM', content)  # 纯数字
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
