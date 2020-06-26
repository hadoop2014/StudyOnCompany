#!/usr/bin/env Python
# coding=utf-8
#本文计划实现对网页财经上的上市公司财务报表中某个特定财务数据的抓取，例如历年的应收票据，全部抓取后存放到excel文件中。

# coding=utf-8
import tushare as ts
#import talib as ta
import numpy as np
import pandas as pd
import os, time, sys, re, datetime
import csv
import scipy
#import re, urllib2
import re,urllib3
import urllib.request as urllib
import xlwt
from bs4 import BeautifulSoup


# 获取股票列表
# code,代码 name,名称 industry,所属行业 area,地区 pe,市盈率 outstanding,流通股本 totals,总股本(万) totalAssets,总资产(万)liquidAssets,流动资产
# fixedAssets,固定资产 reserved,公积金 reservedPerShare,每股公积金 eps,每股收益 bvps,每股净资 pb,市净率 timeToMarket,上市日期
def Get_Stock_List():
    df = ts.get_stock_basics()
    return df


#主要抓取函数在下面，要分析数据在网页上的呈现方式进而选择合适的抓取方式。网易股票的资产负债表的应收票据的数据其实被拆成了2张表，第一张表是纯表头，第二张表是纯数据。

# 抓取网页数据
def Get_3_Cell(url, code, count):
    headers = {"User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6"}
    req = urllib.Request(url, headers=headers)
    try:
        content = urllib.urlopen(req).read()
    except:
        return
    soup = BeautifulSoup(content)

    # 所以要在第一张表中先找到应收票据的位置。
    table0 = soup.find("table", {"class": "table_bg001 border_box limit_sale"})
    j = -1
    for row in table0.findAll("tr"):
        j += 1
        cells = row.findAll("td")
        if len(cells) > 0:  #
            if cells[0].text.find(u'应收票据') >= 0:
                position = j
                # print position
                break

    # 然后到第二张表中去抓对应位置的数据。
    lencell = 0
    table = soup.find("table", {"class": "table_bg001 border_box limit_sale scr_table"})

    j = -1
    for row in table.findAll("tr"):
        cells = row.findAll("td")
        j += 1
        if j == position:
            if len(cells) > 0:  #
                i = 0
                lencell = len(cells)  # 统计财务报表的年数
                while i < len(cells):
                    # print cells[i].text
                    ws.write(count, i + 2, cells[i].text)
                    i = i + 1
        break

    return lencell


def GetData(df_Code, count):
    for Code in df_Code.index:
        print(u"股票代码:" + Code)
        Name = df_Code.loc[Code, 'name']
        print(Name)
        ws.write(count, 0, Code)
        ws.write(count, 1, Name)

        # 资产负债表
        Url1 = 'http://quotes.money.163.com/f10/zcfzb_' + Code + '.html?type=year'
        LenCell1 = Get_3_Cell(Url1, Code, count)
        wb.save('Get3Data1.xls')

        '''
        #利润表
        Url2 = 'http://quotes.money.163.com/f10/lrb_'+Code+'.html?type=year'
        (NumCount,LenCell) = Get_Num(Url2,Code,count)
        wb.save('Get3Data2.xls')
        #现金流量表
        Url3 = 'http://quotes.money.163.com/f10/xjllb_'+Code+'.html?type=year'
        (NumCount,LenCell) = Get_Num(Url3,Code,count)
        wb.save('Get3Data3.xls')
        '''
        #如果到主要财务指标中抓数要到下文进行。
        # Url4 = 'http://quotes.money.163.com/f10/zycwzb_'+Code+'.html?type=year'
        # LenCell4 = Get_Main_Cell(Url4,Code,count)
        # wb.save('GetMainData.xls')
        count = count + 1


#另一个主要抓取函数在下面，要分析数据在网页上的呈现方式进而选择合适的抓取方式。这个是抓取主要财务指标中的资产负债率的数据。

# 抓取网页数据
def Get_Main_Cell(url, code, count):
    headers = {"User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6"}
    req = urllib.Request(url, headers=headers)
    try:
        content = urllib.urlopen(req).read()
    except:
        return
    soup = BeautifulSoup(content)
    # 先找到有资产负债率的那张表（网页上有多个表）

    tables = soup.findAll("table", {"class": "table_bg001 border_box fund_analys"})

    for table in tables:
        # 此处替换中文可修改成获取任意财务数据
        if table.find('td', text=re.compile(u'资产负债率')):
            for row in table.findAll("tr"):
                cells = row.findAll("td")
                if len(cells) > 0:  #
                    j = 1
                    lencell = len(cells)
                years = lencell - 1  # 统计财务报表的年数

                if cells[0].text.find(u'资产负债率') >= 0:
                    # 找到有资产负债率的tr行，然后把td中的数字抓取出来写入excel文件。
                    #print cells[0].text
                    while j < lencell:
                        # print cells[j].text
                        ws.write(count, j+1, cells[j].text)
                        j = j+1


    return years

# 主函数
df = Get_Stock_List()
count = 1
if __name__ == '__main__':
    # 定义excel表格内容
    wb = xlwt.Workbook()
    ws = wb.add_sheet(u'统计表')
    ws.write(0, 0, u'股票代码')
    ws.write(0, 1, u'股票名称')
    ws.write(0, 2, u'2015')
    ws.write(0, 3, u'2014')
    ws.write(0, 4, u'2013')
    ws.write(0, 5, u'2012')
    ws.write(0, 6, u'2011')
    ws.write(0, 7, u'2010')
    ws.write(0, 8, u'2009')
    ws.write(0, 9, u'2008')
    ws.write(0, 10, u'2007')
    ws.write(0, 11, u'2006')
    ws.write(0, 12, u'2005')
    ws.write(0, 13, u'2004')
    ws.write(0, 14, u'2003')
    ws.write(0, 15, u'2002')
    ws.write(0, 16, u'2001')
    ws.write(0, 17, u'2000')

    GetData(df, count)