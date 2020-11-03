#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/20/2020 5:03 PM
# @Author  : wu.hao
# @File    : crawlFinance.py
# @Note    : 用于从互联网上爬取财务报表
import random
import requests
import math
import csv
import pandas as pd

from interpreterCrawl.webcrawl.crawlBaseClass import *


class CrawlFinance(CrawlBase):
    def __init__(self,gConfig):
        super(CrawlFinance, self).__init__(gConfig)
        self.checkpointfilename = os.path.join(self.working_directory, gConfig['checkpointfile'])
        self.checkpointIsOn = self.gConfig['checkpointIsOn'.lower()]
        self.checkpoint = None
        self.checkpointWriter = None


    def crawl_finance_data(self,website,scale):
        assert website in self.dictWebsites.keys(),"website(%s) is not in valid set(%s)"%(website,self.dictWebsites.keys())
        if scale == '批量':
            assert ('公司简称' in self.gConfig.keys() and self.gConfig['公司简称'] != NULLSTR) \
                   and ('报告时间' in self.gConfig.keys() and self.gConfig['报告时间'] != NULLSTR) \
                   and ('报告类型' in self.gConfig.keys() and self.gConfig['报告类型'] != NULLSTR) \
                , "parameter 公司简称(%s) 报告时间(%s) 报告类型(%s) is not valid parameter" \
                  % (self.gConfig['公司简称'], self.gConfig['报告时间'], self.gConfig['报告类型'])

        downloadPaths = self._process_fetch_download_path(website)

        standardPaths = self._process_path_standardize(downloadPaths, website)

        resultPaths = self._process_download(standardPaths, website)

        self.save_checkpoint(resultPaths)

        self.close_checkpoint()


    def _process_fetch_download_path(self,website):
        downloadList = list()
        query_path = self.dictWebsites[website]['query_path']
        headers = self.dictWebsites[website]["headers"]
        query = self.dictWebsites[website]['query']
        query.update({'seDate': self._sedate_transfer(self.gConfig['报告时间'])})
        query.update({'category': self._category_transfer(self.gConfig['报告类型'], website)})
        companys = self.gConfig['公司简称']
        assert isinstance(companys,list),"Company(%s) is not valid,it must be a list!"%companys
        RESPONSE_TIMEOUT = self.dictWebsites[website]['RESPONSE_TIMEOUT']
        exception = self.dictWebsites[website]['exception']
        pageSize = query['pageSize']
        for company in companys:
            query['searchkey'] = company
            query['stock'] = ""
            if company in exception.keys():
                query['stock'] = ','.join([exception[company]['stock'],exception[company]['secid']])
                query['searchkey'] = exception[company]['searchkey']
                #query = {'pageNum': page,  # 页码
                #         'pageSize': 30,
                #         'tabName': 'fulltext',
                #         'column': 'szse',  # 深交所
                #         'stock': '603027,9900024904',  #
                #         'searchkey': '',  # 千禾味业
                #         'secid': '',
                #         'plate': 'sz;sh',
                #         'category': 'category_ndbg_szsh;',  # 年度报告
                #         'trade': '',
                #         'seDate': '2020-01-01~2020-04-26',  # 时间区间
                #         "isHLtitle": 'true'
                #         }
            headers['User-Agent'] = random.choice(self.dictWebsites[website]["user_agent"])  # 定义User_Agent
            try:
                query_response = self._get_response(query_path,headers=headers,data=query,RESPONSE_TIMEOUT=RESPONSE_TIMEOUT)
                time.sleep(random.random() * self.dictWebsites[website]['WAIT_TIME'])
                recordNum = query_response["totalRecordNum"]
                download = query_response['announcements']
                if recordNum > pageSize:
                     endPage = int(math.ceil(recordNum / pageSize))
                     for pageNum in range(1,endPage):
                         query['pageNum'] = pageNum + 1
                         query_response = self._get_response(query_path, headers=headers, data=query,
                                                             RESPONSE_TIMEOUT=RESPONSE_TIMEOUT)
                         time.sleep(random.random() * self.dictWebsites[website]['WAIT_TIME'])
                         recordNum = query_response["totalRecordNum"]
                         download = download +  query_response['announcements']
                downloadList = downloadList + download
                if len(download)  == recordNum:
                    self.logger.info('success to fetch the record num of query response of %s is %d!'%(company,recordNum))
                else:
                    self.logger.warning('failed to fetch total record num of %s : total(%d),fetched(%d)'%(company,recordNum,len(download)))
            except Exception as e:
                self.logger.warning('failed to fetch %s where %s!'%(company,str(e)))
        return downloadList


    def _process_download(self, urllists, website):
        resultPaths = []
        reportType = NULLSTR
        publishingTime = NULLSTR
        for filename,url in urllists.items():
            try:
                path = self._get_path_by_filename(filename)
                #source_directory = os.path.join(self.source_directory,path)
                filePath = os.path.join(path,filename)
                publishingTime = self._get_publishing_time(url)
                reportType = os.path.split(path)[-1]
                resultPaths.append([filename, reportType, publishingTime, url])
                if os.path.exists(filePath):
                    self.logger.info("File %s is already exists!" % filename)
                    continue
                response = requests.get(url)
                file = open(filePath, "wb")
                file.write(response.content)
                file.close()
                self.logger.info("Sucess to fetch %s ,write to file %s!"%(url,filename))
            except Exception as e:
                #如果下载不成功,则去掉该记录
                resultPaths.remove([filename, reportType, publishingTime, url])
                self.logger.error("Failed to fetch %s,file %s!"%(url,filename))
        return resultPaths


    def _process_path_standardize(self, downloadPaths, website):
        assert isinstance(downloadPaths,list),'download paths (%s) is not a list!'%downloadPaths
        download_path = self.dictWebsites[website]['download_path']
        standardPaths = dict()
        for path in downloadPaths:
            urlPath = download_path + path["adjunctUrl"]
            filename = '（' + path["secCode"] + '）' + self._secname_transfer(path['secName']) + '：' \
                       + self._title_transfer(path['announcementTitle']) + '.PDF'
            if '*' in filename:
                filename = filename.replace('*', NULLSTR)
            if self._is_file_needed(filename,website):
                standardPaths.update({filename:urlPath})
        return standardPaths


    def _get_publishing_time(self,url):
        #获取年报的发布时间
        assert url != NULLSTR,"url must not be NULL!"
        publishingTime = NULLSTR
        pattern = self.gJsonInterpreter['TIME']
        matched = self._standardize(pattern, url)
        if matched is not None:
            pulishingTime = matched
        else:
            self.logger.warning('failed to fetch pulishing time of url(%s)'%url)
        return pulishingTime


    def _is_file_needed(self,fileName,website):
        isFileNeeded = True
        if fileName == NULLSTR:
            isFileNeeded = False
        nameDiscard = self.dictWebsites[website]['nameDiscard']
        if nameDiscard != NULLSTR:
            pattern = '|'.join(nameDiscard)
            if self._is_matched(pattern,fileName):
                isFileNeeded = False
        return isFileNeeded


    def _get_response(self,query_path,headers,data,RESPONSE_TIMEOUT):
        query_response = dict()
        self.logger.info('now send query %s!'%query_path)
        try:
            namelist = requests.post(query_path, headers=headers, data=data,timeout=RESPONSE_TIMEOUT)
        except Exception as e:
            self.logger.warning(e)
            return query_response

        if namelist.status_code == requests.codes.ok and namelist.text != '':
            query_response = namelist.json()
        return query_response


    def _title_transfer(self,title):
        timereport = NULLSTR
        pattern = self.gJsonInterpreter['TIME']+self.gJsonInterpreter['VALUE']+ '(（[\\u4E00-\\u9FA5]+）)*'
        matched = self._standardize(pattern,title)
        if matched is not None:
            timereport = matched
        else:
            self.logger.error('title(%s) is error!'%title)
        return timereport


    def _secname_transfer(self, secName):
        company = secName
        #pattern =  "[\\u4E00-\\u9FA5]+"
        pattern = self.gJsonInterpreter['VALUE']
        matched = re.findall(pattern,secName)
        if matched is not None:
            company = ''.join(matched)
        #全角字符转换成半角字符，比如把 华侨城Ａ 转换成 华侨城A
        company = self._strQ2B(company)
        return company


    def _category_transfer(self,category,website):
        assert isinstance(category,list),"category(%s) is invalid!"%category
        dictCategory = self.dictWebsites[website]['category']
        categoryTrans = ';'.join([dictCategory[key] for key in category if key in dictCategory.keys()])
        return categoryTrans


    def _sedate_transfer(self, timelist):
        assert isinstance(timelist,list) and self._is_matched('\\d+年',timelist[0]) and self._is_matched('\\d+年',timelist[-1])\
            ,"timelist(%s) must be a list"%timelist
        seData = timelist[0].split('年')[0] + '-01-01' + '~' + timelist[-1].split('年')[0] + '-12-30'
        return seData


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
    parser=CrawlFinance(gConfig)
    parser.initialize()
    return parser
