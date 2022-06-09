#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:rossliang
# create time:2020/7/10 5:40 下午

from sklearn.cluster import DBSCAN


def cluster(features, algorithm='dbscan'):
    if algorithm == 'dbscan':
        clst = DBSCAN().fit(features)
    else:
        raise NotImplementedError('algorithm %s not support.' % algorithm)
    return clst.label_


if __name__ == '__main__':
    pass
