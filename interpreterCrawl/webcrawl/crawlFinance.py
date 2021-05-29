#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/20/2020 5:03 PM
# @Author  : wu.hao
# @File    : crawlFinance.py
# @Note    : 用于从互联网上爬取财务报表
import random
import requests
import math
import itertools
from time import strptime
from interpreterCrawl.webcrawl.crawlBaseClass import *

class SqliteFinance(SqliteCrawlBase):

    def _write_to_sqlite3(self, dataFrame:DataFrame, commonFields, tableName):
        if dataFrame.shape[0] == 0:
            return
        conn = self._get_connect()
        sql_df = dataFrame.copy()
        # 大立科技2015年有两个年报,（002214）大立科技：2015年年度报告（更新后）.PDF和（002214）大立科技：2015年年度报告（已取消）.PDF,都在2016-03-26发布,需要去掉一个
        sql_df.drop_duplicates(['公司代码','报告类型','报告时间','发布时间'],keep='first',inplace=True)
        # 对于财报发布信息, 必须用报告时间, 报告类型作为过滤关键字
        specialKeys = ['报告类型','发布时间']
        isRecordExist = self._is_record_exist(conn, tableName, sql_df, commonFields, specialKeys=specialKeys)
        if isRecordExist:
            condition = self._get_condition(sql_df, commonFields,specialKeys=specialKeys)
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


class CrawlFinance(CrawlBase):
    def __init__(self,gConfig):
        super(CrawlFinance, self).__init__(gConfig)
        self.database = self.create_database(SqliteFinance)


    def import_finance_data(self,tableName,scale):
        assert tableName in self.tableNames,"website(%s) is not in valid set(%s)"%(tableName,self.tableNames)
        if scale == '批量':
            assert ('公司简称' in self.gConfig.keys() and self.gConfig['公司简称'] != NULLSTR) \
                   and ('报告时间' in self.gConfig.keys() and self.gConfig['报告时间'] != NULLSTR) \
                   and ('报告类型' in self.gConfig.keys() and self.gConfig['报告类型'] != NULLSTR) \
                , "parameter 公司简称(%s) 报告时间(%s) 报告类型(%s) is not valid parameter" \
                  % (self.gConfig['公司简称'], self.gConfig['报告时间'], self.gConfig['报告类型'])
            self._process_import_to_sqlite3(tableName)
        else:
            self.logger.error('now only support scale 批量, but scale(%s) is finded'%scale)


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

        self.checkpoint.save(resultPaths, self.dictWebsites[website]['checkpointHeader'],
                             self.dictWebsites[website]['drop_duplicate'],
                             self.dictWebsites[website]['orderBy'])

        self.checkpoint.close()


    def _process_import_to_sqlite3(self, tableName, encoding = 'utf-8'):
        assert tableName in self.tableNames, "tableName(%s) must be in table list(%s)" % (tableName, self.tableNames)
        checkpoint_content = self.checkpoint.get_content()
        if len(checkpoint_content) == 0:
            return

        #checkpointHeader = self.dictWebsites[website]['checkpointHeader']
        checkpoint_content = [item.split(',') for item in checkpoint_content]
        columnsName = self._get_merged_columns(tableName)
        dataFrame = pd.DataFrame(checkpoint_content, columns=columnsName)
        #dataFrame.columns = self._get_merged_columns(tableName)
        companys = self.gConfig['公司简称']
        reportTypes = self.gConfig['报告类型']
        dataFrameNeeded = dataFrame[dataFrame['公司简称'].isin(companys)]
        dataFrameNeeded = dataFrameNeeded[dataFrameNeeded['报告类型'].isin(reportTypes)]
        #website = '巨潮资讯网'
        #dictCompanys = self._get_deduplicate_companys(companys, self.gConfig['报告时间'], self.gConfig['报告类型'],website)
        for reportType in self.gConfig['报告类型']:
            dataFrameReportType = dataFrameNeeded[dataFrameNeeded['报告类型'] == reportType]
            self.database._write_to_sqlite3(dataFrameReportType, self.commonFields, tableName)
        return


    def _process_save_to_sqlite3(self,fileList, website, encoding = 'utf-8'):
        assert isinstance(fileList,list),"fileList must be a list!"
        tableName = self.dictWebsites[website]['tableName']
        assert tableName in self.tableNames, "tableName(%s) must be in table list(%s)!" % (tableName, self.tableNames)
        if len(fileList) > 0:
            dataFrame = pd.DataFrame(fileList)
            dataFrame.columns = self._get_merged_columns(tableName)
            for reportType in self.gConfig['报告类型']:
                dataFrameReportType = dataFrame[dataFrame['报告类型'] == reportType]
                self.database._write_to_sqlite3(dataFrameReportType, self.commonFields, tableName)


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
        dictTimeToMarkets = self._get_time_to_market(urllists)
        for filename,url in urllists.items():
            try:
                path = self._get_path_by_filename(filename)
                if path == NULLSTR:
                    self.logger.info('the filename (%s) is invalid!'% filename)
                    continue
                #source_directory = os.path.join(self.source_directory,path)
                filePath = os.path.join(path,filename)
                publishingTime = self._get_publishing_time(url)
                reportType = os.path.split(path)[-1]
                company, time, type, code = self._get_company_time_type_code_by_filename(filename)
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
                dictTimeToMarkets[(code,reportType)] = NULLSTR
                self.logger.error("Failed to fetch %s,file %s: %s!"%(url,filename,e))
        resultPaths = self._merge_time_to_market(resultPaths, dictTimeToMarkets)
        return resultPaths


    def _process_path_standardize(self, downloadPaths, website):
        assert isinstance(downloadPaths,list),'download paths (%s) is not a list!'%downloadPaths
        download_path = self.dictWebsites[website]['download_path']
        standardPaths = dict()
        dictPathSize = dict()
        for path in downloadPaths:
            try:
                urlPath = download_path + path["adjunctUrl"]
                code = path["secCode"]
                company = self._secname_transfer(path['secName'])
                type = self._title_transfer(path['announcementTitle'],company)
                filename = '（' + code + '）' + company + '：' \
                           + type + '.PDF'
                publishingTime = self._get_publishing_time(path["adjunctUrl"])
                filename = self._adjust_filename(filename,publishingTime)
                filename = self._get_filename_alias(filename)
                adjunctSize = path['adjunctSize']
                if '*' in filename:
                    filename = filename.replace('*', NULLSTR)
                if self._is_file_needed(filename,website):
                    if filename in dictPathSize.keys():
                        if adjunctSize > dictPathSize[filename]:
                            # （603886）元祖股份：2020年第一季度报告.PDF 有两个第一季度报告,取字节数大的
                            standardPaths.update({filename: urlPath})
                            dictPathSize.update({filename: adjunctSize})
                            self.logger.info('filename(%s) has adjuctSize(%d) is replaced by whitch has adjuctSize(%s)!'
                                             % (filename,dictPathSize[filename],adjunctSize))
                        else:
                            self.logger.info('filename(%s) has adjuctSize(%d) is discarded by whitch has adjuctSize(%s)!'
                                             % (filename,adjunctSize, dictPathSize[filename]))
                    else:
                        standardPaths.update({filename:urlPath})
                        dictPathSize.update({filename: adjunctSize})
            except Exception as e:
                print(e)
                self.logger.error("something is error:urlPath(%s),secCode(%s),secName(%s),announcementTitle(%s)"
                                  %(path["adjunctUrl"], path["secCode"], path['secName'], path['announcementTitle']))
        return standardPaths


    def _adjust_filename(self,filename,publishingTime):
        adjustedFilename = filename
        if publishingTime == NULLSTR:
            return adjustedFilename
        #company,time,type,code = self._get_company_time_type_code_by_filename(filename)
        time = self._get_time_by_filename(filename)
        if time is NaN:
            time = str(strptime(publishingTime,'%Y-%m-%d').tm_year) + '年'
            code, name = filename.split('：')
            #name = self._get_report_type_alias(name)
            adjustedFilename = code +  '：' + time + name
        return adjustedFilename


    def _merge_time_to_market(self, resultPaths, dictTimeToMarkets):
        """
        args:
            resultPaths - 财报文件下载列表,包括如下:
            '''
            "报告时间","公司代码","公司简称","报告类型","发布时间","文件名","网址"
            '''
            dictTimeToMarkets - 公司上市时间:
            '''
            "公司代码","报告类型","上市时间"
            '''
        reutrn:
            resultPaths - 文件下载列表,增加了一个字段 上市时间,规则如下:
            '''
            1) resultPaths和dictTimeToMarkets转换为dataFrame,然后通过 code,type两个字段进行左连接, 将上市时间这个字段加到resultPaths中,最终写到checkpoint文件中.
            '''
        """
        if len(dictTimeToMarkets) == 0:
            return resultPaths
        dictTimeToMarkets = dict([(key, value) if value > self._get_crawl_start_time(key[1]) and value != NULLSTR else (key, NULLSTR)
                                  for key, value in dictTimeToMarkets.items()])
        dataFrameTimeToMarket = pd.DataFrame([[*key, value] for key,value in dictTimeToMarkets.items()], columns=['code', 'type', 'time'])
        dataFrameResultPath = pd.DataFrame(resultPaths,columns=['','code','','type','','',''])
        dataFrameMerged = pd.merge(dataFrameResultPath,dataFrameTimeToMarket,how='left',on=['code','type'])
        dataFrameMerged = dataFrameMerged.fillna(value = NULLSTR)
        resultPaths = dataFrameMerged.values.tolist()
        return resultPaths


    def _get_crawl_start_time(self,reportType):
        """
            args:
                reportType - 报告类型,如下:
                '''
                年度报告
                第一季度报告
                办年度报告
                第三季度报告
                '''
            reutrn:
                crawlStartTime - 爬虫能爬到的年报的最早时间,规则如下:
                '''
                1) 如果是 年度报告, 如果设定爬取2015年年报, 实际可以爬到 2014年的年报, 所以 crawlStartTime 要比设置时间减少一年
                2) 如果是 第一季度报告,半年度报告,第三季度报告, 如果设定爬取2015年年报,实际爬到的也是2015年年报, 此时crawlStartTime不处理
                '''
        """
        crawlStartTime = self.gConfig['报告时间'][0]
        crawlStartTime = self._get_crawl_time(crawlStartTime,reportType)
        return crawlStartTime


    def _get_crawl_time(self,crawlTime, reportType):
        if reportType == '年度报告':
            crawlTime = utile.year_plus(crawlTime, -1)
        return crawlTime


    def _get_time_to_market(self,urllists):
        """
        args:
            urllists - 财报下载列表,包括如下:
            '''
            "报告时间","公司代码","公司简称","报告类型","发布时间","文件名","网址"
            '''
        reutrn:
            dictTimToMarkets - 爬虫能爬到的年报的最早时间,规则如下:
            '''
            1) 其key: value结构为: ('公司代码','报告类型') : '上市时间'
            2) 把urllists按照 ['公司代码','报告类型']进行聚合, 找出最早的时间作为上市时间
            '''
        """
        dictTimToMarkets = dict()
        if len(urllists) == 0:
            return dictTimToMarkets
        urllists = [list(self._get_company_time_type_code_by_filename(filename)) for filename, _ in urllists.items()]
        dataFrame = pd.DataFrame(urllists,columns=['company', 'time', 'type', 'code'])
        dataFrame = dataFrame.groupby(['code','type'])['time'].min()
        dictTimToMarkets = dict(dataFrame)
        return dictTimToMarkets


    def _get_deduplicate_companys(self, companys, reportTime, reportType,website):
        # checkpointfile中已经有的company剔除掉
        # 因为从巨潮咨询网上下载年报数据时,2020年只能下载到2019年的,所以在这里报告时间要进行-1处理
        companysConstruct = []
        for type in reportType:
            #reportTimeList = [self._year_plus(time, -1) for time in reportTime]
            reportTimeList = [self._get_crawl_time(time, type) for time in reportTime]
            companysList = itertools.product(companys,reportTimeList,[type])
            companysConstruct += [','.join(item) for item in companysList]
        companysRequired = self._remove_companys_in_checkpoint(companysConstruct, website)
        companysDiff = set(companysConstruct).difference(set(companysRequired))
        if len(companysDiff) > 0:
            self.logger.info('%d companys is already fetched from %s, no need to process!'
                             % (len(companysDiff),website))
        # 只留下公司,上报类型, 比如: 千禾味业 年度报告. 并且做去重处理
        companysResult = [','.join([item.split(',')[0],item.split(',')[-1]]) for item in companysRequired]
        companysResult = [item.split(',') for item in set(companysResult)]
        # 把相同公司的 上报类型 组合成列表,比如 千禾味业, ['年度报告','第一季度报告']
        dictCompanys = {}
        for company,reportType in companysResult:
            dictCompanys.setdefault(company,[]).append(reportType)
        return dictCompanys


    def _remove_companys_in_checkpoint(self,companysConstruct, website):
        """
        args:
            companysConstruct - 财报下载列表,包括如下:
            '''
            1) 元素由字符串构成,如: "公司简称,报告时间,报告类型"
            '''
            website - 网站名称, 用于dictWebsites配置表的索引
        reutrn:
            companysRequired - 文件下载列表,剔除了在checkpoint文件中记录的已经下载过的文件,剔除规则如下:
            '''
            1) 读取checkpoint文件内容, 拼接出元素为 "公司简称,报告时间,报告类型"的集合, 把这部分记录从companysConstruct中剔除
            2) 读取checkpoint文件中的 上市时间字段, 把把这部分记录从companysConstruct中剔除中报告时间 < 上市时间的记录剔除,
               比如: 执行器要求下载2015-2020年报,但是该公司在2017年上市,则companysConstruct中对应 公司代码,报告类型中 的2015年,2016年的记录被剔除
            '''
        """
        companysRequired = companysConstruct
        #checkpoint = self.get_checkpoint()
        checkpoint_content = self.checkpoint.get_content()
        if len(checkpoint_content) == 0:
            return companysRequired
        checkpointHeader = self.dictWebsites[website]['checkpointHeader']
        checkpoint = [item.split(',') for item in checkpoint_content]
        dataFrame = pd.DataFrame(checkpoint,columns=checkpointHeader)
        dataFrameTimeToMarket = dataFrame[dataFrame['上市时间'] != NULLSTR][['公司简称','报告类型','上市时间']].drop_duplicates()
        dataFrame = dataFrame[['公司简称','报告时间','报告类型']]
        companysCheckpoint = [','.join(item) for item in dataFrame.values.tolist()]
        companysRequired = set(companysConstruct).difference(set(companysCheckpoint))
        companysRequired = set([company for company in companysRequired ])
        dataFrameCompanysRequired = pd.DataFrame([company.split(',') for company in companysRequired],columns=['公司简称','报告时间','报告类型'])
        dataFrameCompanysRequired = pd.merge(dataFrameCompanysRequired,dataFrameTimeToMarket,how='left',on=['公司简称','报告类型'])
        # 对于没有关联上的记录, 上市时间为 NaN,需要替换为NULLSTR,使得下一句的条件判断生效
        dataFrameCompanysRequired = dataFrameCompanysRequired.fillna(value=NULLSTR)
        dataFrameCompanysRequired['上市时间'].replace('nan',NULLSTR,inplace=True)
        dataFrameCompanysRequired = dataFrameCompanysRequired[dataFrameCompanysRequired['报告时间'] > dataFrameCompanysRequired['上市时间']]
        companysRequired = set([','.join(item) for item in dataFrameCompanysRequired[['公司简称','报告时间','报告类型']].values.tolist()])
        return companysRequired


    def _get_publishing_time(self,url):
        #获取年报的发布时间
        assert url != NULLSTR,"url must not be NULL!"
        publishingTime = NULLSTR
        pattern = self.gJsonInterpreter['TIME']
        matched = self._standardize(pattern, url)
        if matched is not NaN:
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
            if utile._is_matched(pattern,fileName):
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


    def _title_transfer(self,title,company):
        timereport = title
        #title = re.sub('<.*>([\\u4E00-\\u9FA5])<.*>', '\g<1>', title)  # 内蒙一机2020年第一季度报告, title中出现 '2020年第<em>一</em>季度'
        title = re.sub('<em>',NULLSTR,title)
        title = re.sub('</em>',NULLSTR,title)
        #title = title.replace('_',NULLSTR) # 解决 ST刚泰 2020年第三季度报告,出现:600687_2020年_三季度报告
        title = re.sub('(\\d{4}年)_三季度报告','\g<1>第三季度报告',title) # 解决 ST刚泰 2020年第三季度报告,出现:600687_2020年_三季度报告
        title = re.sub('(\\d{4}年)_第三季度报告','\g<1>第三季度报告',title) # 解决 （600436）片仔癀：2020年第三季度报告,出现:2020年_第三季度报告
        #title = re.sub('(?!第)三季度报告全文','第三季度报告全文',title) #解决 金博股份：三季度报告全文.PDF
        pattern = self.gJsonInterpreter['TIME']+self.gJsonInterpreter['VALUE']+ '(（[\\u4E00-\\u9FA5]+）)*'
        matched = self._standardize(pattern,title)
        if matched is not NaN:
            timereport = matched
            #time = self._get_time_by_filename(timereport)
            #reportType = self._get_report_type_by_filename(timereport) # 解决 ST刚泰 2020年第三季度报告,出现:600687_2020年_三季度报告
            #reportType = self._get_report_type_alias(reportType)
            #if reportType != NULLSTR:
            #    timereport = time + reportType # 解决 ST刚泰 2020年第三季度报告,出现:600687_2020年_三季度报告
        else:
            self.logger.error('%s title(%s) is error,the right one must like XXXX年(年度报告|第一季度报告|半年度报告|第三季度报告)!'
                              %(company, title))
        return timereport


    def _secname_transfer(self, secName):
        company = secName
        #pattern =  "[\\u4E00-\\u9FA5]+"
        pattern = self.gJsonInterpreter['VALUE']
        matched = re.findall(pattern,secName)
        if matched is not None:
            company = ''.join(matched)
        #全角字符转换成半角字符，比如把 华侨城Ａ 转换成 华侨城A
        company = utile.strQ2B(company)
        company = self._get_company_alias(company)
        return company


    def _category_transfer(self,category,website):
        assert isinstance(category,list),"category(%s) is invalid!"%category
        dictCategory = self.dictWebsites[website]['category']
        categoryTrans = ';'.join([dictCategory[key] for key in category if key in dictCategory.keys()])
        return categoryTrans


    def _sedate_transfer(self, timelist):
        assert isinstance(timelist,list) and utile._is_matched('\\d+年',timelist[0]) and utile._is_matched('\\d+年',timelist[-1])\
            ,"timelist(%s) must be a list"%timelist
        seData = timelist[0].split('年')[0] + '-01-01' + '~' + timelist[-1].split('年')[0] + '-12-30'
        return seData


    def initialize(self,dictParameter = None):
        if dictParameter is not None:
            self.gConfig.update(dictParameter)
        self.loggingspace.clear_directory(self.loggingspace.directory)


def create_object(gConfig):
    parser=CrawlFinance(gConfig)
    parser.initialize()
    return parser
