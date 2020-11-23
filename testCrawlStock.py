# 导入需要使用到的模块
import urllib.request
import re
import pandas as pd
import os


# 爬虫抓取网页函数
def getHtml(url):
    html = urllib.request.urlopen(url).read()
    #html = html.decode('gbk')
    html = html.decode('utf-8')
    return html


# 抓取网页股票代码函数
def getStackCode(html):
    s = r'<li><a target="_blank" href="http://quote.eastmoney.com/\S\S(.*?).html">'
    pat = re.compile(s)
    code = pat.findall(html)
    return code


#########################开始干活############################
Url = 'http://quote.eastmoney.com/stocklist.html'  # 东方财富网股票数据连接地址
Url = 'http://quote.eastmoney.com/center/gridlist.html'
filepath = 'working_directory\\'  # 定义数据文件保存路径
# 实施抓取
html = getHtml(Url)
code = getStackCode(html)
# 获取所有股票代码（以6开头的，应该是沪市数据）集合
CodeList = []
for item in code:
    if item[0] == '6':
        CodeList.append(item)
# 抓取数据并保存到本地csv文件
for code in CodeList:
    print('正在获取股票%s数据' % code)
    url = 'http://quotes.money.163.com/service/chddata.html?code=0' + code + \
          '&end=20161231&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
    urllib.request.urlretrieve(url, filepath + code + '.csv')

##########################将股票数据存入数据库###########################
