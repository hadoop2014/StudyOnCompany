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
import itertools
from interpreterCrawl.webcrawl.crawlBaseClass import *


class CrawlFinance(CrawlBase):
    def __init__(self,gConfig):
        super(CrawlFinance, self).__init__(gConfig)


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

        self._process_save_to_sqlite3(resultPaths, website)

        self.save_checkpoint(resultPaths, website)

        self.close_checkpoint()


    def _process_save_to_sqlite3(self,fileList, website, encoding = 'utf-8'):
        assert isinstance(fileList,list),"fileList must be a list!"
        tableName = self.dictWebsites[website]['tableName']
        assert tableName in self.tableNames, "tableName(%s) must be in table list(%s)!" % (tableName, self.tableNames)
        if len(fileList) > 0:
            dataFrame = pd.DataFrame(fileList)
            dataFrame.columns = self._get_merged_columns(tableName)
            self._write_to_sqlite3(dataFrame, tableName)


    def _write_to_sqlite3(self, dataFrame:DataFrame,tableName):
        conn = self._get_connect()
        sql_df = dataFrame
        # 对于财报发布信息, 必须用报告时间, 报告类型作为过滤关键字
        specialKeys = ['发布时间', '报告类型']
        isRecordExist = self._is_record_exist(conn, tableName, sql_df, specialKeys=specialKeys)
        if isRecordExist:
            condition = self._get_condition(sql_df)
            sql = ''
            sql = sql + 'delete from {}'.format(tableName)
            sql = sql + '\n where ' + condition
            self._sql_executer(sql)
            self.logger.info("delete from {} where is {} {} {}!".format(tableName,sql_df['公司简称'].values[0]
                                                                        ,sql_df['报告时间'].values[0],sql_df['报告类型'].values[0]))
            sql_df.to_sql(name=tableName, con=conn, if_exists='append', index=False)
            conn.commit()
            self.logger.info("insert into {} where is {} {} {}!".format(tableName, sql_df['公司简称'].values[0]
                                                                        ,sql_df['报告时间'].values[0],sql_df['报告类型'].values[0]))
        else:
            sql_df.to_sql(name=tableName,con=conn,if_exists='append',index=False)
            conn.commit()
            self.logger.info("insert into {} where is {} {} {}!".format(tableName, sql_df['公司简称'].values[0]
                                                                            ,sql_df['报告时间'].values[0],sql_df['报告类型'].values[0]))
        conn.close()


    def _process_fetch_download_path(self,website):
        downloadList = list()
        query_path = self.dictWebsites[website]['query_path']
        headers = self.dictWebsites[website]["headers"]
        query = self.dictWebsites[website]['query']
        query.update({'seDate': self._sedate_transfer(self.gConfig['报告时间'])})
        #query.update({'category': self._category_transfer(self.gConfig['报告类型'], website)})
        companys = self.gConfig['公司简称']
        assert isinstance(companys,list),"Company(%s) is not valid,it must be a list!"%companys
        RESPONSE_TIMEOUT = self.dictWebsites[website]['RESPONSE_TIMEOUT']
        exception = self.dictWebsites[website]['exception']
        pageSize = query['pageSize']
        dictCompanys = self._get_deduplicate_companys(companys, self.gConfig['报告时间'], self.gConfig['报告类型'],website)
        for company, reportType in dictCompanys.items():
            query['searchkey'] = company
            query.update({'category': self._category_transfer(reportType, website)})
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
        time, code, company = NULLSTR, NULLSTR, NULLSTR
        for filename,url in urllists.items():
            try:
                path = self._get_path_by_filename(filename)
                #source_directory = os.path.join(self.source_directory,path)
                filePath = os.path.join(path,filename)
                publishingTime = self._get_publishing_time(url)
                reportType = os.path.split(path)[-1]
                company, time, type, code = self._get_time_type_company_code_by_name(filename)
                resultPaths.append([time, code, company, reportType, publishingTime, filename, url])
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
                resultPaths.remove([time, code, company, reportType, publishingTime, filename, url])
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


    def _get_deduplicate_companys(self, companys, reportTime, reportType,website):
        # checkpointfile中已经有的company剔除掉
        checkpoint = self.get_checkpoint()
        checkpointHeader = self.dictWebsites[website]['checkpointHeader']
        checkpoint = [item.split(',') for item in checkpoint]
        dataFrame = pd.DataFrame(checkpoint,columns=checkpointHeader)
        dataFrame = dataFrame[['公司简称','报告时间','报告类型']]
        companysCheckpoint = [','.join(item) for item in dataFrame.values.tolist()]
        # 因为从巨潮咨询网上下载年报数据时,2020年只能下载到2019年的,所以在这里报告时间要进行-1处理
        reportTime = [self._year_plus(time, -1) for time in reportTime]
        companysConstruct = itertools.product(companys,reportTime,reportType)
        companysConstruct = [','.join(item) for item in companysConstruct]
        companysRequired = set(companysConstruct).difference(set(companysCheckpoint))
        companysDiff = set(companysConstruct).difference(set(companysRequired))
        if len(companysDiff) > 0:
            self.logger.info('these companys is already fetched from %s, no need to process: \n%s'
                             % (website, '\n\t'.join(list(sorted(companysDiff)))))
        # 只留下公司,上报类型, 比如: 千禾味业 年度报告. 并且做去重处理
        companysResult = [','.join([item.split(',')[0],item.split(',')[-1]]) for item in companysRequired]
        companysResult = [item.split(',') for item in set(companysResult)]
        # 把相同公司的 上报类型 组合成列表,比如 千禾味业, ['年度报告','第一季度报告']
        dictCompanys = {}
        for company,reportType in companysResult:
            dictCompanys.setdefault(company,[]).append(reportType)
        return dictCompanys


    def _get_publishing_time(self,url):
        #获取年报的发布时间
        assert url != NULLSTR,"url must not be NULL!"
        publishingTime = NULLSTR
        pattern = self.gJsonInterpreter['TIME']
        matched = self._standardize(pattern, url)
        if matched is not None:
            publishingTime = matched
        else:
            self.logger.warning('failed to fetch pulishing time of url(%s)'%url)
        return publishingTime


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
        company = self._get_company_alias(company)
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
