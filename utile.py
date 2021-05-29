#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/25/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserBaseClass.py

# 用于存放一些基础函数
from six import unichr
import time
import datetime
import re
from constant import *
from datetime import date,timedelta,datetime

def get_time_now():
    return time.strftime('%Y%m%d')

def get_today():
    return datetime.now().strftime('%Y-%m-%d')

def time_add(unit, startTime, deltaTime):
    # 支持两种日期格式, 2020-1-30, 2020/1/30
    try:
        startTime = datetime.strptime(startTime, '%Y-%m-%d')
    except:
        startTime = datetime.strptime(startTime, '%Y/%m/%d')
    if unit == 'weeks':
        timeAdded = startTime + timedelta(weeks=deltaTime)
    elif unit == 'days':
        timeAdded = startTime + timedelta(days=deltaTime)
    elif unit == 'hours':
        timeAdded = startTime + timedelta(hours=deltaTime)
    elif unit == 'minutes':
        timeAdded = startTime + timedelta(minutes=deltaTime)
    elif unit == 'seconds':
        timeAdded = startTime + timedelta(seconds=deltaTime)
    elif unit == 'millliseconds':
        timeAdded = startTime + timedelta(milliseconds=deltaTime)
    else:
        raise ValueError('unit(%s) is not supported, now only support unit: weeks,days,hours,minutes,seconds,milliseoconds')
    return timeAdded


def time_difference(unit, startTime, endTime):
    # 支持两种日期格式, 2020-1-30, 2020/1/30
    try:
        endTime = datetime.strptime(endTime,'%Y-%m-%d')
    except:
        endTime = datetime.strptime(endTime,'%Y/%m/%d')
    try:
        startTime = datetime.strptime(startTime,'%Y-%m-%d')
    except:
        startTime = datetime.strptime(startTime, '%Y/%m/%d')
    #if unit == 'year':
    #    timeInterval = rrule.rrule(freq=rrule.YEARLY, dtstart=startTime, until=endTime).count()
    if unit == 'days':
        timeInterval = (endTime - startTime).days
    elif unit == 'seconds':
        # total_seconds是包含天数差,转化过来的秒差
        timeInterval = (endTime - startTime).total_seconds()
    else:
        raise ValueError('unit(%s) is not supported, now only support unit: days,seconds')
    return timeInterval


def get_last_week_day():
    now = date.today()
    if now.isoweekday() == 7:
        dayStep = 2
    elif now.isoweekday() == 6:
        dayStep = 1
    else:
        dayStep = 0
    #print(dayStep)
    lastWorkDay = now - timedelta(days=dayStep)
    lastWorkDay = lastWorkDay.strftime('%Y%m%d')
    return lastWorkDay


def year_plus(reportTime, plusNumber):
    # 2019年 + 1 = 2020年, 2020年 - 1 = 2019年
    assert isinstance(reportTime,str) and isinstance(plusNumber,int),'reportTime must be str and plusNumber must be int!'
    newYear = int(reportTime.split('年')[0]) + plusNumber
    newYear = str(newYear) + '年'
    return newYear


def strQ2B(ustring):
    """把字符串全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
            rstring += uchar
        else:
            rstring += unichr(inside_code)
    return rstring


def strB2Q(ustring):
    """把字符串半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
            inside_code = 0x3000
        else:
            inside_code += 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
            rstring += uchar
        else:
            rstring += unichr(inside_code)
    return rstring


def get_function_explain(function) -> str:
    '''
    explain : 从函数的__doc__中取出函数的用途说明,及explain后的一段文字(只取第一行)
    '''
    func_doc = function.__doc__.strip()
    func_doc = func_doc.split(':')
    func_doc_dict = dict((func_doc[index - 1], word) for index, word in enumerate(func_doc) if index % 2 == 1 )
    func_explain = func_doc_dict.get('explain', NULLSTR)
    func_explain = func_explain.strip().split('\n')[0]
    return func_explain

def _is_matched(pattern,field):
    isMatched = False
    if isinstance(field, str) and isinstance(pattern, str) and pattern != NULLSTR:
        matched = re.search(pattern, field)
        if matched is not None:
            isMatched = True
    return isMatched