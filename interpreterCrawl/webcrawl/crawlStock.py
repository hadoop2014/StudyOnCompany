#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 11/20/2020 5:03 PM
# @Author  : wu.hao
# @File    : crawlStock.py
# @Note    : 用于从互联网上爬取财务报表,被爬取的网站为股城网

import requests
from bs4 import BeautifulSoup
import urllib.request
#import xlwt  # 引入xlwt库，对Excel进行操作。
import csv
import random
from interpreterCrawl.webcrawl.crawlBaseClass import *


class CrawlStock(CrawlBase):
    def __init__(self,gConfig):
        super(CrawlStock, self).__init__(gConfig)
        #self.checkpointfilename = os.path.join(self.working_directory, gConfig['checkpointfile'])
        #self.checkpointIsOn = self.gConfig['checkpointIsOn'.lower()]
        #self.checkpoint = None
        #self.checkpointWriter = None


    def crawl_stock_data(self,website,scale):
        assert website in self.dictWebsites.keys(),"website(%s) is not in valid set(%s)"%(website,self.dictWebsites.keys())
        #stockList = []
        resultPaths = []
        if scale == '批量':
            assert ('公司简称' in self.gConfig.keys() and self.gConfig['公司简称'] != NULLSTR) \
                   and ('报告时间' in self.gConfig.keys() and self.gConfig['报告时间'] != NULLSTR) \
                   and ('报告类型' in self.gConfig.keys() and self.gConfig['报告类型'] != NULLSTR) \
                , "parameter 公司简称(%s) 报告时间(%s) 报告类型(%s) is not valid parameter" \
                  % (self.gConfig['公司简称'], self.gConfig['报告时间'], self.gConfig['报告类型'])
            #stockList = self._process_stock_list(website)
            #self._save_stock_list(stockList)
            stockList = self._get_stock_list(self.gConfig['公司简称'])
            # 把指数数据同时也下载了
            indexList = self._get_index_list(self.indexes)
            stockList = stockList + indexList
            resultPaths = self._process_fetch_stock_data(stockList, website)
            resultPaths = self._process_save_to_sqlite3(resultPaths, website, encoding='gbk')
        elif scale == '全量':
            stockList = self._process_stock_list(website)
            self._save_stock_list(stockList)
        #resultPaths = self._process_fetch_stock_data(stockList, website)
        #resultPaths = self._process_save_to_sqlite3(resultPaths, website, encoding='gbk')
        self.save_checkpoint(resultPaths, website)
        self.close_checkpoint()


    def _process_save_to_sqlite3(self,fileList, website, encoding = 'utf-8'):
        assert isinstance(fileList, list),"fileList must be a list!"
        tableName = self.dictWebsites[website]['tableName']
        assert tableName in self.tables, "tableName(%s) must be in table list(%s)"% (tableName, self.tables)
        successPaths = []
        for fileName in fileList:
            fullfileName = os.path.join(self.working_directory, fileName[0])
            if not os.path.exists(fullfileName):
                self.logger.info('file %s is not exist!'% fullfileName)
                continue
            dataFrame = pd.read_csv(fullfileName, encoding = encoding)
            dataFrame.columns = self._get_merged_columns(tableName)
            if not dataFrame.empty:
                self._write_to_sqlite3(dataFrame,tableName)
                successPaths.append(fileName)
            else:
                self.logger.info('failed to write to sqlite3,the file is empty: %s'% fileName)
            #self.logger.info("success to write to sqlite3 from file %s"% fullfileName)
        return successPaths


    def _process_fetch_stock_data(self, stockList, website):
        stockInfoURL = self.dictWebsites[website]['download_path']
        tableName = self.dictWebsites[website]['tableName']
        fieldNameEn = self.dictTables[tableName]['fieldAlias'].values()
        endTime = time.strftime('%Y%m%d')
        resultPaths = []
        stockList = self._get_deduplicate_stock(stockList,endTime)
        for company, code, type in stockList:
            codeTransfer = self._code_transfer(code, type)
            if codeTransfer == NULLSTR:
                continue
            url = stockInfoURL \
                  + '?code=' + codeTransfer \
                  + '&end=' + endTime \
                  + '&fields=' + ';'.join(fieldNameEn)
            fileName = "（" + code + "）" + company + '.csv'
            fullfileName = os.path.join(self.working_directory, fileName)
            try:
                if os.path.exists(fullfileName):
                    os.remove(fullfileName)
                #headers['User-Agent'] = random.choice(self.dictWebsites[website]["user_agent"])
                urllib.request.urlretrieve(url, fullfileName)
                resultPaths.append([fileName, company, str(code), endTime])
                time.sleep(random.random() * self.dictWebsites[website]['WAIT_TIME'])
                self.logger.info('success to fetch stock (%s)%s trading data!'% (code, company))
            except Exception as e:
                print(e)
                self.logger.error('failed to fetch stock (%s)%s trading data from %s!'% (code, company, url))
        companyDiffer = set([company for company,_,_ in stockList]).difference([company for _,company,_,_ in resultPaths])
        if len(companyDiffer) > 0:
            self.logger.info("failed to fetch stock data : %s"% companyDiffer)
        return resultPaths


    def _get_index_list(self,indexes):
        assert isinstance(indexes,list),"indexes(%s) must be a list!" % indexes
        indexList = []
        for index in indexes:
            if index in self.gJsonBase['stockindex'].keys():
                code = self.gJsonBase['stockindex'][index]
                indexList.append([index, code, "指数"])
        return indexList


    def _get_deduplicate_stock(self,stockList,endTime):
        # 从stockList中去掉checkpoint中已经记录的部分,这部分已经入了数据库,不再需要下载了, 因为交易数据按天更新,所以要考虑endTime
        checkpoint = self.get_checkpoint()
        checkpointList = [','.join(lines.split(',')[1:]) for lines in checkpoint]
        stockDiffer = set([','.join([stock[0],stock[1],endTime]) for stock in stockList]).difference(set(checkpointList))
        stockListResult = []
        for stock in stockList:
            for remainStock in stockDiffer:
                if stock[0] == remainStock.split(',')[0] and remainStock.split(',')[-1] == endTime:
                    # 只有公司名字 和 截止时间都相等的,才加入stockListResult
                    stockListResult.append(stock)
        return stockListResult


    def _code_transfer(self,code, type):
        codeTransfer = NULLSTR
        if code == NULLSTR or type == NULLSTR:
            return codeTransfer
        if type == '公司':
            if list(code)[0] == '6':
                # 所有6打头的股票代码都是沪市的,前面加 0,其他股票代码,前面加 1
                codeTransfer = '0' + str(code)
            else:
                codeTransfer = '1' + str(code)
        elif type == '指数':
            if list(code)[0] == '0':
                # 对于上证指数, 以 0 开头
                codeTransfer = '0' + str(code)
            else:
                # 对于深市指出, 以 1 开头
                codeTransfer = '1' + str(code)
        else:
            # 如果是国债,则直接返回空
            pass
        return codeTransfer


    def _save_stock_list(self,stockList):
        assert isinstance(stockList, list),"parameter stockList(%s) must be a list!"% stockList
        if os.path.exists(self.stockcodefile):
            os.remove(self.stockcodefile)
        stockList = sorted(stockList,key=lambda x: x[2] + x[1])
        stockcodefile = open(self.stockcodefile, 'w', newline= '', encoding= 'utf-8')
        stockcodefileWriter = csv.writer(stockcodefile)
        stockcodefileWriter.writerows(stockList)
        stockcodefile.close()
        self.logger.info('sucess to write stock code into file %s'% self.stockcodefile)

    '''
    def _process_fetch_stock_info(self,resultPaths, website):
        assert isinstance(resultPaths,list),"resultPaths must be list!"
        stockInfoURL = self.dictWebsites[website]['download_path']
        count = 0
        # lst = [item.lower() for item in lst]  股城网url是大写,所以不用切换成小写
        for stock in resultPaths:
            url = stockInfoURL + stock + "/"  # url为单只股票的url
            html = self._getHTMLText(url)  # 爬取单只股票网页，得到HTML
            try:
                if html == "":  # 爬取失败，则继续爬取下一只股票
                    continue
                infoDict = {}  # 单只股票的信息存储在一个字典中
                soup = BeautifulSoup(html, 'html.parser')  # 单只股票做一锅粥
                stockInfo = soup.find('div', attrs={'class': 'stock_top clearfix'})
                # 在观察股城网时发现，单只股票信息都存放在div的'class':'stock_top clearfix'中
                # 在soup中找到所有标签div中属性为'class':'stock_top clearfix'的内容
                name = stockInfo.find_all(attrs={'class': 'stock_title'})[0]
                # 在stockInfo中找到存放有股票名称和代码的'stock_title'标签
                infoDict["股票代码"] = name.text.split("\n")[2]
                infoDict.update({'股票名称': name.text.split("\n")[1]})
                # 对name以换行进行分割，得到一个列表，第1项为股票名称，第2项为代码
                # 如果以空格股票名称中包含空格，会产生异常，
                # 如“万 科A",得到股票名称为万，代码为科A

                keyList = stockInfo.find_all('dt')
                valueList = stockInfo.find_all('dd')
                # 股票信息都存放在dt和dd标签中，用find_all产生列表
                for i in range(len(keyList)):
                    key = keyList[i].text
                    val = valueList[i].text
                    infoDict[key] = val
                    # 将信息的名称和值作为键值对，存入字典中

                with open(fpath, 'a', encoding='utf-8') as f:
                    f.write(str(infoDict) + '\n')
                    # 将每只股票信息作为一行输入文件中
                    count = count + 1
                    self.logger.info("\r爬取成功，当前进度: {:.2f}%".format(count * 100 / len(resultPaths)), end="")
            except:
                count = count + 1
                self.logger.info("\r爬取失败，当前进度: {:.2f}%".format(count * 100 / len(resultPaths)), end="")
                continue
    '''

    def _process_stock_list(self,website):
        # 获取股票代码列表
        stockURL = self.dictWebsites[website]['query_path']
        stockList = []
        if website == "网易财经":
            #html = self._getHTMLText(stockURL, "GB2312")
            html = self._getHTMLText(stockURL)
            soup = BeautifulSoup(html, 'html.parser')
            a = soup.find_all('a')  # 得到一个列表
            for i in a:
                try:
                    href = i.attrs['href']  # 股票代码都存放在href标签中
                    matched = re.findall(r"[S][HZ]\d{6}", href)
                    if len(matched) > 0:
                        #if i.span is not None:
                            # 对于上证指数, 深证成指按如下处理
                        #    content = i.span.contents[0]
                        #    code, company, type = self._content_transfer(content)
                        #    code = matched[0]
                            #type = '指数'
                        #else:
                        content = i.contents[0]
                        code, company, type = self._content_transfer(content)
                            #type = '公司'
                        if code is not NaN and company is not NaN:
                            # 针对TMT50指数, 在这里去掉
                            stockList.append([company, code, type])
                except:
                    continue
        #elif website == "东方财富网":
        #    html = self._getHtml(website,code='utf-8')
        #    code = self._get_stack_code(html)
            # 获取所有股票代码（以6开头的，应该是沪市数据）集合
            #CodeList = []
        #    for item in code:
        #        if item[0] == '6':
        #            stockList.append(item)
        self.logger.info('success to fetch stock code from %s' % stockURL)
        return stockList


    def _content_transfer(self, content):
        assert content != NULLSTR and isinstance(content, str), "content must not be NULL!"
        type = '公司'
        content = content.replace(' ',NULLSTR).replace('(',"（").replace(')',"）")
        company, code = self._get_company_code_by_content(content)
        if company in self.gJsonBase['stockindex'].keys():
            type = '指数'
        else:
            matched = re.search('国债\\d*',company)
            if matched is not None:
                type = '国债'
        return code, company, type


    def _getHTMLText(self,url, code="utf-8"):  # 获取HTML文本
        try:
            r = requests.get(url)
            r.raise_for_status()
            r.encoding = code
            return r.text
        except Exception as e:
            print(e)
            return NULLSTR


    '''
    def _getHtml(self,website,code='gbk'):
        # 爬虫抓取网页函数,东方财富网
        url = self.dictWebsites[website]["query_path"]
        html = urllib.request.urlopen(url).read()
        html = html.decode(code)
        return html
    '''
    '''
    def _get_stack_code(self,html):
        # 抓取网页股票代码函数,东方财富网
        s = r'<li><a target="_blank" href="http://quote.eastmoney.com/\S\S(.*?).html">'
        pat = re.compile(s)
        code = pat.findall(html)
        return code
    '''

    '''
    def save_checkpoint(self, content):
        assert isinstance(content,list),"Parameter content(%s) must be list"%(content)
        content = self._remove_duplicate(content)
        self.checkpoint.seek(0)
        self.checkpoint.truncate()
        self.checkpointWriter.writerows(content)
        #读取checkpoint内容,去掉重复记录,重新排序,写入文件


    def close_checkpoint(self):
        self.checkpoint.close()


    def _remove_duplicate(self,content):
        assert isinstance(content, list), "Parameter content(%s) must be list" % (content)
        resultContent = content
        if len(content) == 0:
            return resultContent
        checkpointHeader = self.gJsonInterpreter['checkpointHeader']
        dataFrame = pd.read_csv(self.checkpointfilename,names=checkpointHeader)
        dataFrame = dataFrame.append(pd.DataFrame(content,columns=checkpointHeader))
        dataFrame = dataFrame.drop_duplicates()
        dataFrame = dataFrame.sort_values(by=["报告类型","文件名"],ascending=False)
        resultContent = dataFrame.values.tolist()
        return resultContent
    '''

    def initialize(self,dictParameter = None):
        if dictParameter is not None:
            self.gConfig.update(dictParameter)
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self.clear_logging_directory(self.logging_directory)
        if self.checkpointIsOn:
            if not os.path.exists(self.checkpointfilename):
                fw = open(self.checkpointfilename,'w',newline='',encoding='utf-8')
                fw.close()
            self.checkpoint = open(self.checkpointfilename, 'r+', newline='', encoding='utf-8')
            self.checkpointWriter = csv.writer(self.checkpoint)
        else:
            if os.path.exists(self.checkpointfilename):
                os.remove(self.checkpointfilename)


def create_object(gConfig):
    parser=CrawlStock(gConfig)
    parser.initialize()
    return parser



