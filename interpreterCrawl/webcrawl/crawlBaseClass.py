#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/25/2019 5:03 PM
# @Author  : wu.hao
# @File    : crawlBaseClass.py

import time
from interpreterCrawl.interpreterBaseClass import *

class SqliteCrawl(SqilteBase):

    def _get_max_min_trading_date(self,conn, tableName, dataFrame, commonFields):
        minTradingDate, maxTradingDate = None, None
        condition = self._get_condition(dataFrame,commonFields)
        sql = "select min(报告时间), max(报告时间) from {} where ".format(tableName) + condition
        try:
            result = conn.execute(sql).fetchall()
            if len(result) > 0:
                minTradingDate, maxTradingDate = result[0]
        except Exception as e:
            print(e)
            self.logger.error('failed to get max & min trading data from sql:%s'% sql)
        return minTradingDate,maxTradingDate

    def _write_to_sqlite3(self, dataFrame:DataFrame, commonFields, tableName):
        conn = self._get_connect()
        sql_df = dataFrame
        sql_df['公司代码'] = dataFrame['公司代码'].apply(lambda x: x.replace('\'', NULLSTR))
        sql_df['公司简称'] = dataFrame['公司简称'].apply(lambda x: x.replace(' ', NULLSTR))
        minTradingDate, maxTradingDate = self._get_max_min_trading_date(conn, tableName, sql_df, commonFields)
        if minTradingDate is not None and maxTradingDate is not None:
            if sql_df['报告时间'].max() > maxTradingDate:
                #self.logger.info("delete from {} where is {} {} - {}!".format(tableName,sql_df['公司简称'].values[0]
                #                                                             ,maxTradingDate, sql_df['报告时间'].max()))
                sql_df = sql_df[sql_df['报告时间'] > maxTradingDate]
                if not sql_df.empty:
                    sql_df.to_sql(name=tableName, con=conn, if_exists='append', index=False)
                    conn.commit()
                    self.logger.info("insert into {} where is {} {} - {}!".format(tableName, sql_df['公司简称'].values[0]
                                                                                 ,sql_df['报告时间'].values[-1]
                                                                                 ,sql_df['报告时间'].values[0]))
        else:
            sql_df.to_sql(name=tableName,con=conn,if_exists='append',index=False)
            conn.commit()
            self.logger.info("insert into {} where is {} {} - {}!".format(tableName, sql_df['公司简称'].values[0]
                                                                        ,sql_df['报告时间'].values[-1],sql_df['报告时间'].values[0]))
        conn.close()

#深度学习模型的基类
class CrawlBase(InterpreterBase):
    def __init__(self,gConfig):
        super(CrawlBase, self).__init__(gConfig)
        self.start_time = time.time()
        #self.model_savefile = os.path.join(self.workingspace.directory, self._get_module_name() + '.model')
        #self.checkpoint_filename = self._get_class_name(self.gConfig) + '.ckpt'
        self.source_directory = os.path.join(self.data_directory, self.gConfig['source_directory'])
        self.sourceFile = os.path.join(self.data_directory, self.gConfig['source_directory'],
                                       self.gConfig['sourcefile'])
        self.checkpointIsOn = self.gConfig['checkpointIsOn'.lower()]
        self.checkpointfilename = os.path.join(self.workingspace.directory, self.gConfig['checkpointfile'])
        self.checkpoint = None
        self.checkpointWriter = None
        self.database = self.create_database(SqliteCrawl)
        #self._create_tables(self.tableNames)
    '''
    def _get_class_name(self, gConfig):
        parser_name = re.findall('Crawl(.*)', self.__class__.__name__).pop().lower()
        return parser_name
    '''

    def save_checkpoint(self, content, website):
        assert isinstance(content,list),"Parameter content(%s) must be list"%(content)
        if len(content) == 0:
            return
        content = self._remove_duplicate(content, website)
        self.checkpoint.seek(0)
        self.checkpoint.truncate()
        self.checkpointWriter.writerows(content)
        #读取checkpoint内容,去掉重复记录,重新排序,写入文件


    def close_checkpoint(self):
        self.checkpoint.close()


    def get_checkpoint(self):
        if self.checkpointIsOn == False:
            return
        with open(self.checkpointfilename, 'r', encoding='utf-8') as csv_in:
            reader = csv_in.read().splitlines()
        return reader


    def _remove_duplicate(self,content,website):
        assert isinstance(content, list), "Parameter content(%s) must be list" % (content)
        resultContent = content
        if len(content) == 0:
            return resultContent
        checkpointHeader = self.dictWebsites[website]['checkpointHeader']
        dataFrame = pd.read_csv(self.checkpointfilename,names=checkpointHeader,dtype=str)
        dataFrame = dataFrame.append(pd.DataFrame(content,columns=checkpointHeader,dtype=str))
        # 根据数据第一列去重
        dataFrame = dataFrame.drop_duplicates(self.dictWebsites[website]['drop_duplicate'], keep= 'last')
        order = self.dictWebsites[website]['orderBy']
        dataFrame = dataFrame.sort_values(by=order, ascending=False)
        resultContent = dataFrame.values.tolist()
        return resultContent

    '''
    def getSaveFile(self):
        if self.model_savefile == NULLSTR:
            self.model_savefile = None
            return None
        if self.model_savefile is not None:
            if os.path.exists(self.model_savefile) == False:
                return None
                # 文件不存在
        return self.model_savefile


    def removeSaveFile(self):
        if self.model_savefile is not None:
            filename = os.path.join(os.getcwd(), self.model_savefile)
            if os.path.exists(filename):
                os.remove(filename)
    '''

    def debug_info(self, info=None):
        if self.debugIsOn == False:
            return
        pass
        return


    def debug(self, layer, name=NULLSTR):
        pass


    def initialize(self):
        self.loggingspace.clear_directory(self.loggingspace.directory)