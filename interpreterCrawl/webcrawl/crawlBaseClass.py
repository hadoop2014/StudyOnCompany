#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/25/2019 5:03 PM
# @Author  : wu.hao
# @File    : crawlBaseClass.py

from interpreterCrawl.interpreterBaseClass import *

class SqliteCrawlBase(SqilteBase):

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


class CheckpointCrawl(CheckpointBase):

    @Multiprocess.lock
    def save(self, content, checkpoint_header, drop_duplicate, order_by):
        assert isinstance(content, list), "Parameter content(%s) must be list" % (content)
        if len(content) == 0 or self.checkpointIsOn == False:
            return
        # 读取checkpoint内容,去掉重复记录,重新排序,写入文件
        checkpoint = self._get_checkpoint()
        checkpointWriter = csv.writer(checkpoint)
        content = self._remove_duplicate(checkpoint, content, checkpoint_header, drop_duplicate, order_by)
        checkpoint.seek(0)
        checkpoint.truncate()
        checkpointWriter.writerows(content)
        checkpoint.close()
        self.logger.info('Success to write checkpoint to file %s' % self.checkpointfile)

    def _remove_duplicate(self, checkpoint, content, checkpoint_header, drop_duplicate, order_by):
        assert isinstance(content, list), "Parameter content(%s) must be list" % (content)
        resultContent = content
        if len(content) == 0 or checkpoint is None:
            return resultContent
        checkpoint.seek(0)
        dataFrame = pd.read_csv(checkpoint, names=checkpoint_header, dtype=str)
        dataFrame = dataFrame.append(pd.DataFrame(content, columns=checkpoint_header, dtype=str))
        # 根据数据第一列去重
        dataFrame = dataFrame.drop_duplicates(drop_duplicate, keep='last')
        dataFrame = dataFrame.sort_values(by=order_by, ascending=False)
        resultContent = dataFrame.values.tolist()
        return resultContent


class CrawlBase(InterpreterBase):
    def __init__(self,gConfig):
        super(CrawlBase, self).__init__(gConfig)
        self.start_time = time.time()
        self.database = self.create_database(SqliteCrawlBase)
        self.checkpoint = self.create_space(CheckpointCrawl)

    def debug_info(self, info=None):
        if self.debugIsOn == False:
            return
        pass
        return


    def debug(self, layer, name=NULLSTR):
        pass


    def initialize(self):
        self.loggingspace.clear_directory(self.loggingspace.directory)