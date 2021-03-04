#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 12/3/2020 5:03 PM
# @Author  : wu.hao
# @File    : companyEvaluate.py
# @Note    : 用于公司价格分析

from interpreterAnalysize.interpreterBaseClass import *


class StockAnalysize(InterpreterBase):
    def __init__(self,gConfig):
        super(StockAnalysize, self).__init__(gConfig)
        self.checkpointIsOn = gConfig['checkpointIsOn'.lower()]


    def stock_index_trend_analysize(self, tableName, scale):
        if scale == '批量':
            assert '指数简称' in self.gConfig.keys() and self.gConfig['指数简称'] != NULLSTR\
                ,'parameter 指数简称(%s) is not valid!' % self.gConfig['指数简称']
            sourceTableName = self.dictTables[tableName]['sourceTable']
            dataFrame = self._sql_to_dataframe(tableName, sourceTableName,scale)
            conn = self._get_connect()
            if self._is_table_exist(conn, tableName):
                self._drop_table(conn, tableName)
            conn.close()
            for indexName in self.gConfig['指数简称']:
                self._index_trend_analysize(dataFrame, indexName,tableName)


    def _index_trend_analysize(self,dataFrame: DataFrame, indexName,tableName):
        """
        explain:
            用于有indexName制定的股票指数, 分析其微趋势,短期趋势,中期趋势,长期趋势
        Example:
            无
        args:
            dataFrame - 从 股票交易数据 中读取的股票指数数据
            indexName - 股票指数名称, 如: 上证指数,深证成值
        return:
            无
        """
        indexCode = self.gJsonBase['stockindex'][indexName]
        dataFrame = dataFrame[dataFrame['公司代码'] == int(indexCode)].loc[:,['公司代码','公司简称','收盘价','报告时间']]
        dataFrame = dataFrame.sort_values(by=['公司代码','报告时间']).reset_index(drop=True)
        dataFrame = dataFrame.reset_index()
        dataFrame.rename(columns = {'index':'recordId'},inplace=True)
        # 扩充三列
        dataFrame['趋势简称'] = NULLSTR
        dataFrame['是否上升趋势'] = None
        dataFrame['isUtmost'] = False   # 用于搜索中期趋势,长期趋势的极点
        # 计算短期,中期,长期移动平均线
        indexMovingAverage = self._compute_moving_average(dataFrame, tableName, indexName)
        # 寻找微趋势, 即连续上涨或连续下跌的天数
        #indexTrendMini = self._find_tiny_trend(dataFrame,indexName)
        # 寻找短期趋势, 即由搜索区间minorTerm设定的值所找到的上涨或下跌趋势
        trendName = '短期趋势'
        minTermDay = self.dictTables[tableName]['trend_settings']['minTermDay']
        minTermRateMinor = self._get_min_term_rate(tableName,indexName,trendName)
        minorTerm = self.dictTables[tableName]['trend_settings']['minorTerm']
        indexTrendMinor = self._find_min_max_date(dataFrame,minorTerm, indexName,trendName)
        indexTrendMinor = self._merge_trend(indexTrendMinor,indexName,trendName)
        indexTrendMinor = self._delete_invalid_trend(indexTrendMinor, minTermDay,minTermRateMinor, indexName, trendName)
        indexTrendMinor = self._merge_trend(indexTrendMinor, indexName, trendName)

        # 寻找中期趋势, 既由搜索区间medianTerm设定的值所找到的上涨或下降趋势
        trendName = '中期趋势'
        minMedianTermRate = self._get_min_term_rate(tableName,indexName,trendName)
        medianTerm = self.dictTables[tableName]['trend_settings']['medianTerm']
        indexTrendMedian = self._find_utmost_in_trend(indexTrendMinor, medianTerm, minMedianTermRate, indexName, trendName)
        indexTrendMedian = self._merge_trend(indexTrendMedian,indexName,trendName)
        indexTrendMedian = self._delete_invalid_trend(indexTrendMedian,minTermDay,minMedianTermRate,indexName,trendName)
        #indexTrendMedian = self._merge_trend(indexTrendMedian, indexName, trendName)
        # 整理所有中期趋势
        trendInfoMedian = self._find_index_trend(indexTrendMedian,tableName, indexName, trendName)

        # 寻找长期期趋势, 既由搜索区间longTerm设定的值所找到的上涨或下降趋势
        trendName = '长期趋势'
        minLongTermRate = self._get_min_term_rate(tableName,indexName,trendName)
        longTerm = self.dictTables[tableName]['trend_settings']['longTerm']
        indexTrendLong = self._find_utmost_in_trend(indexTrendMinor, longTerm, minLongTermRate, indexName, trendName)
        indexTrendLong = self._merge_trend(indexTrendLong,indexName,trendName)
        indexTrendLong = self._delete_invalid_trend(indexTrendLong,minTermDay,minLongTermRate,indexName,trendName)
        #indexTrendLong = self._merge_trend(indexTrendLong, indexName, trendName)
        # 整理所有长期趋势
        trendInfoLong = self._find_index_trend(indexTrendLong, tableName, indexName, trendName)
        # 最后一个长期趋势修正
        trendInfoLong = self._judge_last_long_trend(trendInfoLong, indexMovingAverage, tableName, indexName, trendName)

        trendInfoMerged = pd.concat([trendInfoMedian,trendInfoLong], axis=0)
        self._write_to_sqlite3(trendInfoMerged, tableName)


    def _judge_last_long_trend(self, trendInfo, indexMovingAverage, tableName, indexName, trendName):
        """
        explain:
            根据整理后的趋势数据, 对最后一个长期趋势进行适配,算法:
            1) 判断最后一个长期趋势,即趋势开始到当前时刻在longTerm(默认400个交易日)之内, 以这个趋势为基准趋势, 假定是个上升趋势.
            2) 从最后一个交易日开始计算, 如果下一个趋势结束日的收盘价在长期趋势线(默认为150日移动平均线)之上的, 则归并为上升趋势；反之,则认为趋势反转,应归并为下降趋势
        Example:
            无
        args:
            trendInfo - 整理过后的趋势数据,包括: 趋势涨幅,趋势收盘价
            tableName - 待写入数据库的表名称,为: 指数趋势分析表
            indexName - 股票指数名称, 如: 上证指数,深证成值
            trendName - 趋势名称,为:短期趋势,中期趋势,长期趋势
        return:
            trendInfoLocal - 整理出来的趋势信息, 前后两点统计出一个趋势.
        """
        startTime = time.time()
        assert trendName == '长期趋势','trendName(%s) of %s is invalid,该函数只处理长期趋势!' % (trendName, indexName)
        trendInfoLocal = pd.DataFrame(columns=self.dictTables[tableName]['fieldName'])
        if trendInfo.shape[0] <= 1:
            self.logger.warning('indexTrend may be empty:%s' % trendInfo)
            return trendInfoLocal
        # longTermDay定义长期趋势的最大持续时长
        longTermDay = self.dictTables[tableName]['trend_settings']['longTermDay']

        lastTrendType = NULLSTR
        #iDimension = 0
        today = datetime.now().strftime('%Y-%m-%d')
        for iLoop in range(trendInfo.shape[0]):
            if self._time_difference('days', trendInfo['报告时间'].iloc[iLoop], today) <= longTermDay:
                if lastTrendType == NULLSTR:
                    # 记录满足条件的最后一个趋势类型
                    lastTrendType = trendInfo['趋势类型'].iloc[iLoop]
                    # 放入本地数据结构trendInfoLocal
                    trendInfoLocal = trendInfoLocal.append(trendInfo.iloc[iLoop].copy())
                    #iDimension += 1
                else:
                    # 最后一个长期趋势的归并
                    # 读取长期移动平均数
                    lastDate = self._time_add('days',trendInfo['报告时间'].iloc[iLoop], trendInfo['趋势时长'].iloc[iLoop])
                    lastDate = lastDate.strftime('%Y-%m-%d')
                    lastIndexMovingAvgLong = indexMovingAverage.loc[indexMovingAverage['报告时间'] == lastDate,'长期移动平均线']
                    if lastTrendType == '上升':
                        if trendInfo['结束收盘价'].iloc[iLoop] >= lastIndexMovingAvgLong:
                            # 如果最后一个参考趋势为上升趋势, 且趋势结束日期收盘价在长期趋势线的上方, 则进行归并
                            iDimension = trendInfoLocal.index[-1]
                            trendInfoLocal.loc[iDimension, '结束收盘价'] = trendInfo['结束收盘价'].iloc[iLoop]
                            trendInfoLocal.loc[iDimension, '趋势涨幅'] = round((trendInfoLocal['结束收盘价'].iloc[iDimension]
                                                                            - trendInfoLocal['收盘价'].iloc[iDimension])
                                                                            / trendInfoLocal['收盘价'].iloc[iDimension],2)
                            trendInfoLocal.loc[iDimension, '趋势交易天数'] = trendInfoLocal['趋势交易天数'].iloc[iDimension] \
                                                                         + trendInfo['趋势交易天数'].iloc[iLoop]
                            trendInfoLocal.loc[iDimension, '趋势持续时长'] = trendInfoLocal['趋势持续时长'].iloc[iDimension] \
                                                                         + trendInfo['趋势持续时长'].iloc[iLoop]
                        else:
                            # 说明趋势发生变化
                            # 放入trendInfoLocal
                            lastTrendType = '下降'
                            trendInfoLocal = trendInfoLocal.append(trendInfo.iloc[iLoop].copy())
                            #iDimension += 1
                    else:
                        # 如果最后一个参考趋势是下降趋势
                        if trendInfo['结束收盘价'].iloc[iLoop] <= lastIndexMovingAvgLong:
                            # 如果最后一个参考趋势是下降趋势, 且趋势结束日收盘价在长期趋势线的上方,则进行归并
                            iDimension = trendInfoLocal.index[-1]
                            trendInfoLocal.loc[iDimension, '结束收盘价'] = trendInfo['结束收盘价'].iloc[iLoop]
                            trendInfoLocal.loc[iDimension, '趋势涨幅'] = round((trendInfoLocal['结束收盘价'].iloc[iDimension]
                                                                            - trendInfoLocal['收盘价'].iloc[iDimension])
                                                                            / trendInfoLocal['收盘价'].iloc[iDimension],2)
                            trendInfoLocal.loc[iDimension, '趋势交易天数'] = trendInfoLocal['趋势交易天数'].iloc[iDimension] \
                                                                         + trendInfo['趋势交易天数'].iloc[iLoop]
                            trendInfoLocal.loc[iDimension, '趋势持续时长'] = trendInfoLocal['趋势持续时长'].iloc[iDimension] \
                                                                         + trendInfo['趋势持续时长'].iloc[iLoop]
                        else:
                            # 说明趋势发生变化
                            lastTrendType = '上升'
                            trendInfoLocal = trendInfoLocal.append(trendInfo.iloc[iLoop].copy())
                            #iDimension += 1
            else:
                # 记入本地数据结构
                trendInfoLocal = trendInfoLocal.append(trendInfo.iloc[iLoop].copy())
                #iDimension += 1

        self.logger.info('success to judge last trend info %s of %s : processtime %.4f'
                         % (trendName,indexName, time.time() - startTime))
        return trendInfoLocal


    def _find_index_trend(self,indexTrend,tableName, indexName, trendName):
        """
        explain:
            从indexTrend记录的趋势中, 按照前后两点把趋势整理出来:
        Example:
            无
        args:
            indexTrend - 股票指数的趋势数据,前后两点为趋势的高点/低点, 两点定义一个趋势
            indexName - 股票指数名称, 如: 上证指数,深证成值
            trendName - 趋势名称,为:短期趋势,中期趋势,长期趋势
        return:
            trendInfo - 整理出来的趋势信息, 前后两点统计出一个趋势.
        """
        startTime = time.time()
        trendInfo = pd.DataFrame(columns=self.dictTables[tableName]['fieldName'])
        if indexTrend.shape[0] <= 1:
            self.logger.warning('indexTrend may be empty:%s' % indexTrend)
            return trendInfo
        # trendInfo 第一个点初始化
        trendInfo.loc[0, '报告时间'] = indexTrend['报告时间'].iloc[0]
        trendInfo.loc[0, '公司代码'] = indexTrend['公司代码'].iloc[0]
        trendInfo.loc[0, '公司简称'] = indexTrend['公司简称'].iloc[0]
        trendInfo.loc[0, '收盘价'] = round(indexTrend['收盘价'].iloc[0],4)
        trendInfo.loc[0, '趋势简称'] = indexTrend['趋势简称'].iloc[0]
        trendInfo.loc[0, '趋势起始点'] = indexTrend['recordId'].iloc[0]
        trendInfo.loc[0, '趋势交易天数'] = indexTrend['recordId'].iloc[0]
        trendInfo.loc[0, '趋势类型'] = NULLSTR
        trendInfo.loc[0, '趋势持续时长'] = 0
        trendInfo.loc[0, '趋势涨幅'] = 0
        trendInfo.loc[0, '结束收盘价'] = 0
        trendInfo.loc[0, '训练标识'] = 1

        #trendType = NULLSTR
        iDimension = 0
        # 对邻近两个短期趋势, 其趋势方向相同, 则合并为一个中期趋势
        for iLoop in range(1, indexTrend.shape[0]):
            rate = round((indexTrend['收盘价'].iloc[iLoop] - indexTrend['收盘价'].iloc[iLoop - 1])
                         /indexTrend['收盘价'].iloc[iLoop - 1] , 2)
            if rate >= 0 :
                trendType = '上升'
            else:
                trendType = '下降'

            if trendInfo['趋势类型'].iloc[iDimension] == NULLSTR:
                # 第一次计算
                trendInfo.loc[iDimension,'趋势类型'] = trendType
            elif trendType != trendInfo['趋势类型'].iloc[-1]:
                trendInfo.loc[iDimension,'结束收盘价'] = round(indexTrend['收盘价'].iloc[iLoop - 1],4)
                trendInfo.loc[iDimension,'趋势持续时长'] = self._time_difference('days',trendInfo['报告时间'].iloc[iDimension]
                                                                         , indexTrend['报告时间'].iloc[iLoop - 1])
                trendInfo.loc[iDimension,'趋势涨幅'] = round((trendInfo['结束收盘价'].iloc[iDimension] - trendInfo['收盘价'].iloc[iDimension])
                                                         /trendInfo['收盘价'].iloc[iDimension],2)
                trendInfo.loc[iDimension,'趋势交易天数'] = indexTrend['recordId'].iloc[iLoop - 1] - trendInfo['趋势交易天数'].iloc[iDimension]

                iDimension += 1
                trendInfo.loc[iDimension,'报告时间'] = indexTrend['报告时间'].iloc[iLoop - 1]
                trendInfo.loc[iDimension,'收盘价'] = round(indexTrend['收盘价'].iloc[iLoop - 1],4)
                trendInfo.loc[iDimension,'公司代码'] = indexTrend['公司代码'].iloc[iLoop - 1]
                trendInfo.loc[iDimension,'公司简称'] = indexTrend['公司简称'].iloc[iLoop - 1]
                trendInfo.loc[iDimension,'趋势简称'] = indexTrend['趋势简称'].iloc[iLoop - 1]
                trendInfo.loc[iDimension,'趋势类型'] = trendType
                trendInfo.loc[iDimension,'趋势起始点'] = indexTrend['recordId'].iloc[iLoop - 1]
                trendInfo.loc[iDimension,'趋势交易天数'] = indexTrend['recordId'].iloc[iLoop - 1]
                trendInfo.loc[iDimension,'训练标识'] = 1

        # 处理最后一点
        trendInfo.loc[iDimension, '结束收盘价'] = round(indexTrend['收盘价'].iloc[-1],4)
        trendInfo.loc[iDimension, '趋势持续时长'] = self._time_difference('days', trendInfo['报告时间'].iloc[iDimension]
                                                                  , indexTrend['报告时间'].iloc[-1])
        trendInfo.loc[iDimension, '趋势涨幅'] = round((trendInfo['结束收盘价'].iloc[iDimension] - trendInfo['收盘价'].iloc[iDimension])
                                                     / trendInfo['收盘价'].iloc[iDimension], 2)
        trendInfo.loc[iDimension, '趋势交易天数'] = indexTrend['recordId'].iloc[-1] - trendInfo['趋势交易天数'].iloc[iDimension]
        #if trendName == "中期趋势":
            # 最后一个中期趋势点可能还不是定论,需要通过模型来预测
        trendInfo.loc[iDimension, '训练标识'] = 0
        self.logger.info('success to find trend info %s of %s : processtime %.4f'
                         % (trendName,indexName, time.time() - startTime))
        return trendInfo


    def _find_utmost_in_trend(self,indexTrend, dayOfTerm, minTermRate, indexName, trendName):
        """
        explain:
            从indexTrend记录的趋势中, 按照dayOfTerm定义的搜索区间, 找到趋势的极点, 极点即趋势的转折点,趋势还要符合如下条件:
            1) 趋势的升副或跌幅大于minTermRate
        Example:
            无
        args:
            indexTrend - 股票指数的趋势数据,前后两点为趋势的高点/低点, 两点定义一个趋势
            dayOfTerm - 趋势的搜索时长
            minTermRate - 一个趋势要成立,起涨跌幅必须大于minTermRate
            indexName - 股票指数名称, 如: 上证指数,深证成值
            trendName - 趋势名称,为:短期趋势,中期趋势,长期趋势
        return:
            indexTrendLocal - 删除非法趋势后遗留下来的趋势, 一组最大最小值定义一个趋势,第一点和最后一点必须保留下来.
        """
        startTime = time.time()
        indexTrendLocal = pd.DataFrame(columns=indexTrend.columns)
        if indexTrend.shape[0] <= 1:
            self.logger.warning('indexTrend may be empty:%s' % indexTrend)
            return indexTrendLocal
        indexTrendLocal = indexTrend.copy()
        # 需要把index重设置为'index'
        indexTrendLocal = indexTrendLocal.reset_index(drop=True)
        indexTrendLocal = indexTrendLocal.reset_index(drop=False)
        # 第一个点设置为极点
        indexTrendLocal.loc[0,['isUtmost']] = True
        if indexTrendLocal['收盘价'].iloc[1] >= indexTrendLocal['收盘价'].iloc[0]:
            # 初始趋势为上升趋势
            lastTrendAscend = True
        else:
            # 初始趋势为下降趋势
            lastTrendAscend = False
        iLoop = 0
        maxIndex, minIndex = None,None
        while iLoop < indexTrendLocal.shape[0]:
            minIndex = indexTrendLocal.iloc[iLoop].copy()
            maxIndex = indexTrendLocal.iloc[iLoop].copy()
            iLoop += 1
            for iCount in range(iLoop, indexTrendLocal.shape[0]):
                if indexTrendLocal['recordId'].iloc[iCount] - indexTrendLocal['recordId'].iloc[iLoop] <= dayOfTerm:
                    if indexTrendLocal['收盘价'].iloc[iCount] < minIndex['收盘价']:
                        minIndex = indexTrendLocal.iloc[iCount].copy()
                    if indexTrendLocal['收盘价'].iloc[iCount] > maxIndex['收盘价']:
                        maxIndex = indexTrendLocal.iloc[iCount].copy()
                else:
                    if minIndex['recordId'] > maxIndex['recordId']:
                        # 高点在前, 低点在后, 这是一个下降趋势
                        if lastTrendAscend == True:
                            # 前一个趋势是上升趋势, 说明趋势反转, 反转点即极点.
                            indexTrendLocal.loc[maxIndex['index'],['isUtmost']] = True
                        else:
                            # 说明是延续上一个的下降趋势, 如果这次的高点和上次的低点不是同一个点, 则认为上一个低点和本次高点为极点
                            if maxIndex['index'] != iLoop - 1 \
                                and abs((indexTrendLocal['收盘价'].iloc[maxIndex['index']] - indexTrendLocal['收盘价'].iloc[iLoop - 1])
                                        /indexTrendLocal['收盘价'].iloc[iLoop - 1]) * 100.0 > minTermRate:
                                indexTrendLocal.loc[iLoop - 1, ['isUtmost']] = True
                                indexTrendLocal.loc[maxIndex['index'], 'isUtmost'] = True
                        # 设置当前趋势为下降趋势, 作为下一次搜索的起点
                        lastTrendAscend = False
                        # 从地点开始下一次搜索
                        iLoop = minIndex['index']
                    elif minIndex['recordId'] < maxIndex['recordId']:
                        # 低点在前, 高点在后, 是一个上升趋势
                        if lastTrendAscend == False:
                            # 说明趋势反转, 前一个去是下降趋势, 当前趋势是上升趋势, 则反转点即极点
                            indexTrendLocal.loc[minIndex['index'], ['isUtmost']] = True
                        else:
                            # 说明是上一个上升趋势的延续, 如果这次的低点和上次的高点不是同一个点, 则认为上一个高点是极点
                            if minIndex['index'] != iLoop - 1 \
                                and abs((indexTrendLocal['收盘价'].iloc[minIndex['index']] - indexTrendLocal['收盘价'].iloc[iLoop - 1])
                                        /indexTrendLocal['收盘价'].iloc[iLoop - 1]) * 100.0 > minTermRate:
                                indexTrendLocal.loc[iLoop-1,['isUtmost']] = True
                                indexTrendLocal.loc[minIndex['index'],['isUtmost']] = True
                        lastTrendAscend = True
                        # 从高点开始下一轮搜索
                        iLoop = maxIndex['index']
                    else:
                        # 高点和低点是同一点, 说明本次没有搜索到任何趋势, 则往下走一步
                        iLoop += 1
                    # 跳出内循环
                    break

        # 最后一个趋势的处理, maxIndex比如等于minIndex
        if maxIndex['index'] != indexTrendLocal['index'].iloc[-1]:
            indexTrendLocal.loc[maxIndex['index'], ['isUtmost']] = True
        # 最后一个点设置为极点
        indexTrendLocal.loc[indexTrendLocal['index'].iloc[-1],['isUtmost']] = True

        #长期趋势修正
        for iCount in range(indexTrendLocal.shape[0]):
            if indexName == '上证指数':
                # 上证指数修正, 1992/9/25日算极点
                if self._time_difference('days', indexTrendLocal['报告时间'].iloc[iCount], '1992/5/25') == 0:
                    indexTrendLocal.loc[iCount, ['isUtmost']] = True
                elif self._time_difference('days', indexTrendLocal['报告时间'].iloc[iCount], '1999/5/18') == 0:
                    indexTrendLocal.loc[iCount, ['isUtmost']] = True
                elif self._time_difference('days', indexTrendLocal['报告时间'].iloc[iCount], '1997/9/23') == 0:
                    # 例外情况
                    indexTrendLocal.loc[iCount, ['isUtmost']] = False
            elif indexName == '恒生指数':
                # 恒生指数长期趋势修正
                if self._time_difference('days', indexTrendLocal['报告时间'].iloc[iCount], '2009/3/9') == 0:
                    indexTrendLocal.loc[iCount, ['isUtmost']] = True
                elif self._time_difference('days', indexTrendLocal['报告时间'].iloc[iCount], '2008/10/27') == 0:
                    indexTrendLocal.loc[iCount, ['isUtmost']] = False
            elif indexName == '道琼斯指数':
                # 道琼斯指数长期趋势修正
                if self._time_difference('days', indexTrendLocal['报告时间'].iloc[iCount], '1929/9/3') == 0:
                    indexTrendLocal.loc[iCount, ['isUtmost']] = True

        # 对数据进行归整
        indexTrendLocal = indexTrendLocal.drop(columns= 'index')
        indexTrendLocal['趋势简称'] = trendName
        indexTrendLocal = indexTrendLocal.set_index(['recordId'], drop=False)
        self.logger.info('success to delete invalid %s of %s : processtime %.4f'
                         % (trendName,indexName, time.time() - startTime))
        return indexTrendLocal


    def _delete_invalid_trend(self,indexTrend, minTermDay, minTermRate, indexName, trendName):
        """
        explain:
            从indexTrend记录的趋势中, 删除不合法的趋势,合法的趋势有两点:
            1) 趋势的持续时长大于minTermDay.
            2) 趋势的升副或跌幅大于minTermRate
            3) 趋势的极点必须保留, 即isUtmost = True
        Example:
            无
        args:
            indexTrend - 股票指数的趋势数据,前后两点为趋势的高点/低点, 两点定义一个趋势
            minTermDay - 一个趋势要成立,起持续时长必须大于minTermDay
            minTermRate - 一个趋势要成立,起涨跌幅必须大于minTermRate
            indexName - 股票指数名称, 如: 上证指数,深证成值
            trendName - 趋势名称,为:短期趋势,中期趋势,长期趋势
        return:
            indexTrendLocal - 删除非法趋势后遗留下来的趋势, 一组最大最小值定义一个趋势,第一点和最后一点必须保留下来.
        """
        startTime = time.time()
        indexTrendLocal = pd.DataFrame(columns=indexTrend.columns)
        if indexTrend.shape[0] <= 1:
            self.logger.warning('indexTrend may be empty:%s' % indexTrend)
            return indexTrendLocal
        # 记录第一点
        indexTrendLocal = indexTrendLocal.append(indexTrend.iloc[0].copy())
        for recordId in range(1, indexTrend.shape[0] - 1):
            rate = round((indexTrend['收盘价'].iloc[recordId + 1] - indexTrend['收盘价'].iloc[recordId])/
                         indexTrend['收盘价'].iloc[recordId] * 100, 2)
            if abs(rate) > minTermRate or indexTrend['isUtmost'].iloc[recordId] == True:
                indexTrendLocal = indexTrendLocal.append(indexTrend.iloc[recordId].copy())

        # 最后一点的处理
        if indexTrendLocal['recordId'].iloc[-1] != indexTrend['recordId'].iloc[-1]:
            indexTrendLocal = indexTrendLocal.append(indexTrend.iloc[-1].copy())

        self.logger.info('success to delete invalid %s of %s : processtime %.4f'
                         % (trendName,indexName, time.time() - startTime))
        return indexTrendLocal


    def _merge_trend(self,indexTrend, indexName, trendName):
        """
        explain:
            根据indexTrend中的趋势, 把前后两个趋势相同的趋势进行合并, 如前后两点都是上升趋势,则合并成一个趋势
        Example:
            无
        args:
            indexTrend - 股票指数的趋势数据,前后两点为趋势的高点/低点, 两点定义一个趋势
            indexName - 股票指数名称, 如: 上证指数,深证成值
            trendName - 趋势名称,为:短期趋势,中期趋势,长期趋势
        return:
            indexTrend - 对前后两个相同趋势进行合并后的趋势数据, 一组最大最小值定义一个趋势,第一点和最后一点必须保留下来.
        """
        startTime = time.time()
        indexTrendLocal = pd.DataFrame(columns=indexTrend.columns)
        if indexTrend.shape[0] <= 1:
            self.logger.warning('indexTrend may be empty:%s' % indexTrend)
            return indexTrendLocal
        # 记录第一个点
        indexTrendLocal = indexTrendLocal.append(indexTrend.iloc[0].copy())
        isLastTrendAscend = indexTrend['是否上升趋势'].iloc[0]
        for recordId in range(1, indexTrend.shape[0]):
            #rate = round((indexTrend['收盘价'].iloc[recordId] - indexTrend['收盘价'].iloc[recordId - 1])
            #             / indexTrend['收盘价'].iloc[recordId - 1],2)
            #if rate == 0:
            #    indexTrendLocal = indexTrendLocal.append(indexTrend.iloc[recordId].copy())
            #    lastRate = rate
            #elif rate * lastRate < 0:
            #    indexTrendLocal = indexTrendLocal.append(indexTrend.iloc[recordId].copy())
            #    lastRate = rate
            if indexTrend['是否上升趋势'].iloc[recordId] != isLastTrendAscend:
                indexTrendLocal = indexTrendLocal.append(indexTrend.iloc[recordId].copy())
                isLastTrendAscend = indexTrendLocal['是否上升趋势'].iloc[-1]

        if indexTrendLocal['recordId'].iloc[-1] != indexTrend['recordId'].iloc[-1]:
            # 记录最后一个点
            indexTrendLocal = indexTrendLocal.append(indexTrend.iloc[-1].copy())
        self.logger.info('success to merge %s of %s : processtime %.4f'
                         % (trendName,indexName, time.time() - startTime))
        return indexTrendLocal


    def _find_min_max_date(self,dataFrame:DataFrame, dayOfTerm, indexName, trendName):
        """
        explain:
            根据搜索时长dayOfTerm,找到指数历史数据中的最大最小值,前后两个时间点确定一组趋势, 为下一步做趋势合并等做准备
        Example:
            无
        args:
            dataFrame - 从 股票交易数据 中读取的股票指数数据
            indexName - 股票指数名称, 如: 上证指数,深证成值
            dayOfTerm - 定义趋势的搜索时长, 可以是短期趋势时长,中期趋势时长,长期趋势时长
            trendName - 趋势名称,可为: 短期趋势, 中期趋势, 长期趋势
        return:
            indexTrend - 记录当前指数的最大最小值数据, 一组最大最小值定义一个趋势. 其中第一点和最后一点都会放入趋势中
        """
        startTime = time.time()
        indexTrend = pd.DataFrame(columns=dataFrame.columns)
        if dataFrame.shape[0] <= 1:
            self.logger.warning('dataFrame may be empty:%s' % dataFrame)
            return indexTrend
        recordId = 0
        maxIndex = dataFrame.iloc[recordId].copy()
        minIndex = dataFrame.iloc[recordId].copy()
        indexTrend = indexTrend.append(dataFrame.iloc[recordId].copy())
        #for recordId in range(dataFrame.shape[0]):
        recordId += 1
        while recordId < dataFrame.shape[0]:
            if recordId - indexTrend['recordId'].iloc[-1] < dayOfTerm:
                # 计算有效交易日的区间 在搜索门限之内
                if dataFrame['收盘价'].iloc[recordId] < minIndex['收盘价']:
                    minIndex = dataFrame.iloc[recordId].copy()
                if dataFrame['收盘价'].iloc[recordId] > maxIndex['收盘价']:
                    maxIndex = dataFrame.iloc[recordId].copy()
            else:
                # 已经超出了一个搜索区间, 可以判断上一个区间是一个什么趋势
                if self._time_difference(unit ='days',startTime = maxIndex['报告时间'], endTime=minIndex['报告时间']) > 0:
                    # 最大值的时间在前, 最小值的时间在后, 这是一个下降趋势
                    if self._time_difference('days',indexTrend['报告时间'].iloc[-1],maxIndex['报告时间']) != 0 :
                        # 最大值不等于上一次趋势的终点, 表示已经搜索到一个完整周期的上升趋势, 则保留上次搜索到的这个完整趋势
                        if maxIndex['收盘价'] > indexTrend['收盘价'].iloc[-1]:
                            # 上一个趋势修正为上升趋势
                            indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = True
                        else:
                            indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = False
                        indexTrend = indexTrend.append(maxIndex.copy())
                        indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = False
                    else:
                        # 最大值就是上一次趋势的终点,这种情况下,需要继续向前搜索, 起始点设置为最小值那一天
                        indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = False
                        maxIndex = minIndex.copy()
                        indexTrend = indexTrend.append(minIndex.copy())
                        #indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = False
                    recordId = indexTrend['recordId'].iloc[-1]
                elif self._time_difference('days',maxIndex['报告时间'], minIndex['报告时间']) < 0:
                    # 最小值的时间在前,最大值的时间在后,这是一个上升趋势
                    if self._time_difference('days', indexTrend['报告时间'].iloc[-1], minIndex['报告时间']) != 0:
                        # 最小值不是上一次趋势终点, 说明已经搜索到一个完整周期的下降趋势, 则保留上次搜索到的这个完整趋势
                        if minIndex['收盘价'] > indexTrend['收盘价'].iloc[-1]:
                            # 上一个趋势修正为上升趋势
                            indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = True
                        else:
                            indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = False
                        indexTrend = indexTrend.append(minIndex.copy())
                        indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = True
                    else:
                        # 最小值就是上一次趋势终点,这种情况下,需要继续向前搜索, 起始点设置为最大值的那一天
                        indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = True
                        minIndex = maxIndex.copy()
                        indexTrend = indexTrend.append(maxIndex.copy())
                        #indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = True
                    recordId = indexTrend['recordId'].iloc[-1]
                else:
                    # 如果maxIndex['发布日期'] == minIndex['发布日期'], 说明出现了问题, 则跳过这段时间, 上证指数从1999年的2月9日直接跳到2009年的3月1日
                    indexTrend.iloc[-1] = dataFrame.iloc[recordId].copy()
            recordId += 1

        # 最后一个趋势的处理
        if self._time_difference('days',maxIndex['报告时间'], minIndex['报告时间']) > 0 :
            # 最大值在前, 最小值在后, 这是一个下降趋势
            if self._time_difference('days', indexTrend['报告时间'].iloc[-1], maxIndex['报告时间']) != 0:
                # 最大值不等于上一次趋势的终点, 表示已经搜索到一个完整周期的上升趋势, 则保留上次搜索到的这个完整趋势
                if maxIndex['收盘价'] > indexTrend['收盘价'].iloc[-1]:
                    # 上一个趋势修正为上升趋势
                    indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = True
                else:
                    indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = False
                indexTrend = indexTrend.append(maxIndex.copy())
                indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = False
            else:
                # 最大值就是上一次趋势的终点,这种情况下,需要继续向前搜索, 起始点设置为最小值那一天
                indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = False
                #maxIndex = minIndex.copy()
                indexTrend = indexTrend.append(minIndex.copy())
        elif self._time_difference('days',maxIndex['报告时间'], minIndex['报告时间']) < 0:
            # 最小值在前, 最大值在后, 这是一个上升趋势
            if self._time_difference('days', indexTrend['报告时间'].iloc[-1], minIndex['报告时间']) != 0:
                # 最小值不是上一次趋势终点, 说明已经搜索到一个完整周期的下降趋势, 则保留上次搜索到的这个完整趋势
                if minIndex['收盘价'] > indexTrend['收盘价'].iloc[-1]:
                    # 上一个趋势修正为上升趋势
                    indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = True
                else:
                    indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = False
                indexTrend = indexTrend.append(minIndex.copy())
                indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = True
            else:
                # 最小值就是上一次趋势终点,这种情况下,需要继续向前搜索, 起始点设置为最大值的那一天
                indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = True
                #minIndex = maxIndex.copy()
                indexTrend = indexTrend.append(maxIndex.copy())
        else:
            # 最大值和最小值在同一点,无需处理
            pass

        # 把最后一个趋势到最后一天也记录到趋势中
        if self._time_difference('days',indexTrend['报告时间'].iloc[-1], dataFrame['报告时间'].iloc[-1]) != 0:
            if dataFrame['收盘价'].iloc[-1] >= indexTrend['收盘价'].iloc[-1]:
                # 最后一个趋势到最后一个点是上升趋势
                indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = True
            else:
                indexTrend.loc[indexTrend.index[-1], '是否上升趋势'] = False
            indexTrend = indexTrend.append(dataFrame.iloc[-1].copy())

        indexTrend['趋势简称'] = trendName
        self.logger.info('success to find all %s of %s : processtime %.4f'
                         % (trendName,indexName, time.time() - startTime))
        return indexTrend


    def _find_tiny_trend(self,dataFrame: DataFrame,indexName):
        """
        explain:
            用于有查找指数中的微趋势, 微趋势定义为: 连续上升(包括平价) 或 连续下降的 点形成一个上升或下降的微趋势.
            第一点和最后一点必须列入
        Example:
            无
        args:
            dataFrame - 指定指数名称的股票指数数据: '公司代码', '公司简称', '收盘价', '报告时间', '趋势简称', '是否上升趋势'
        return:
            indexTrend - 记录当前指数的所有微趋势, 趋势简称统一设置为 微趋势
        """
        startTime = time.time()
        indexTrend = pd.DataFrame(columns=dataFrame.columns)
        if dataFrame.shape[0] <= 1:
            self.logger.warning('dataFrame may be empty:%s' % dataFrame)
            return indexTrend
        indexTrend = dataFrame.iloc[[0,1]].copy()
        if indexTrend['收盘价'].iloc[1] >= indexTrend['收盘价'].iloc[0]:
            # 第一个趋势是上升趋势, 平盘也计算为上升趋势
            isLastTrendAscend = True
        else:
            # 第一个趋势是下降趋势
            isLastTrendAscend = False
        indexTrend.loc[indexTrend.index[0],'是否上升趋势'] = isLastTrendAscend
        for recordId in range(dataFrame.shape[0]):
            if recordId <= 1:
                continue
            if dataFrame['收盘价'].iloc[recordId] >= indexTrend['收盘价'].iloc[-1]:
                # 当前股价在上升趋势中
                if isLastTrendAscend == False:
                    # 如果前一个趋势是下降趋势, 说明当前点是一个转折点, 表明一个新的上升趋势开始, 则记录当前点
                    indexTrend.loc[indexTrend.index[-1],'是否上升趋势'] = isLastTrendAscend
                    indexTrend = indexTrend.append(dataFrame.iloc[recordId].copy())
                else:
                    # 如果前一个去是是上升趋势, 则用新指覆盖上一点的值
                    #indexTrend = indexTrend.drop(indexTrend.index[-1])
                    #indexTrend = indexTrend.append(dataFrame.iloc[recordId].copy())
                    indexTrend.iloc[-1] = dataFrame.iloc[recordId].copy()
                    #indexTrend.loc[indexTrend.index[-1]] = dataFrame.iloc[iRecordId].values.tolist()
                isLastTrendAscend = True
            else:
                # 当前股价处于下降趋势中
                if isLastTrendAscend == True:
                    # 如果前一个去是是上升去是, 说明当前点是一个转折点, 表明一个新的下降趋势开始, 则记录当前点
                    indexTrend.loc[indexTrend.index[-1],'是否上升趋势'] = isLastTrendAscend
                    indexTrend = indexTrend.append(dataFrame.iloc[recordId].copy())
                else:
                    #indexTrend = indexTrend.drop(indexTrend.index[-1])
                    #indexTrend = indexTrend.append(dataFrame.iloc[recordId].copy())
                    indexTrend.iloc[-1] = dataFrame.iloc[recordId].copy()
                    #indexTrend.loc[indexTrend.index[-1]] = dataFrame.iloc[iRecordId].values.tolist()
                isLastTrendAscend = False
        # 最后一个趋势赋值
        indexTrend.loc[indexTrend.index[-1],'是否上升趋势'] = isLastTrendAscend
        indexTrend['趋势简称'] = '微趋势'
        self.logger.info('success to find all tiny trend of %s : processtime %.4f' % (indexName, time.time() - startTime))
        return indexTrend


    def _get_min_term_rate(self,tableName, indexName, trendName):
        minTermRate = None
        assert indexName in self.gJsonBase['stockindex'].keys(),'indexName(%s) is invalid, it must be in %s' \
                                                                % (indexName, self.gJsonBase['stockindex'].keys())
        if trendName == '短期趋势':
            minTermRate = self.dictTables[tableName]['trend_settings']['minMinorTermRate']
            if indexName == '道琼斯指数':
                minTermRate = self.dictTables[tableName]['trend_settings']['minMinorTermRateDJI']
        elif trendName == '中期趋势':
            minTermRate = self.dictTables[tableName]['trend_settings']['minMedianTermRate']
        elif trendName == '长期趋势':
            minTermRate = self.dictTables[tableName]['trend_settings']['minLongTermRate']
            if indexName == '道琼斯指数':
                minTermRate = self.dictTables[tableName]['trend_settings']['minLongTermRateDJI']
        else:
            self.logger.warning('trend(%s) is invalid, now only support 短期趋势, 中期趋势, 长期趋势!' % trendName)
        return minTermRate


    def _compute_moving_average(self, dataFrame: DataFrame, tableName, indexName):
        """
        explain:
            计算指数的移动平均线,算法:
            1) 根据minorMovingAvgTerm参数,计算短期移动平均线,
            2) 根据medianMovingAvgTerm参数,计算中期移动平均线,
            3) 根据longMovingAvgTerm参数,计算长期移动平均线,
        Example:
            无
        args:
            dataFrame - 保护了indexName指定的所有指数数据
            tableName - 待写入数据库的表名称,为: 指数趋势分析表
            indexName - 股票指数名称, 如: 上证指数,深证成值
        return:
            indexMovingAverage - 记录指数中期,短期,长期移动平均线.
        """
        startTime = time.time()
        indexMovingAverage = pd.DataFrame(columns=dataFrame.columns.values.tolist() + ['短期移动平均线', '中期移动平均线', '长期移动平均线', '价格震荡指数', '累加值'])
        if dataFrame.shape[0] <= 1:
            self.logger.warning('indexTrend may be empty:%s' % dataFrame)
            return indexMovingAverage

        minorMovingAvgTerm = self.dictTables[tableName]['trend_settings']['minorMovingAvgTerm']
        medianMovingAvgTerm = self.dictTables[tableName]['trend_settings']['medianMovingAvgTerm']
        longMovingAvgTerm = self.dictTables[tableName]['trend_settings']['longMovingAvgTerm']
        for iloop in range(dataFrame.shape[0]):
            indexMovingAverage.loc[iloop,dataFrame.columns] = dataFrame.iloc[iloop].copy()
            if iloop == 0:
                indexMovingAverage.loc[iloop, '累加值'] = dataFrame['收盘价'].iloc[iloop]
            else:
                indexMovingAverage.loc[iloop, '累加值'] = indexMovingAverage['累加值'].iloc[iloop - 1] + dataFrame['收盘价'].iloc[iloop]

            # 更新短期(默认30天)移动平均线
            if iloop < minorMovingAvgTerm:
                indexMovingAverage.loc[iloop, '短期移动平均线'] = 0
            else:
                indexMovingAverage.loc[iloop, '短期移动平均线'] = round((indexMovingAverage['累加值'].iloc[iloop]
                                                                  - indexMovingAverage['累加值'].iloc[iloop - minorMovingAvgTerm]) / minorMovingAvgTerm,2)

            # 更新中期(默认150天)移动平均线
            if iloop < medianMovingAvgTerm:
                indexMovingAverage.loc[iloop, '中期移动平均线'] = 0
            else:
                indexMovingAverage.loc[iloop, '中期移动平均线'] = round((indexMovingAverage['累加值'].iloc[iloop]
                                                                    - indexMovingAverage['累加值'].iloc[iloop - medianMovingAvgTerm]) / medianMovingAvgTerm,2)

            # 更新长期(默认250天)移动平均线
            if iloop < longMovingAvgTerm:
                indexMovingAverage.loc[iloop, '长期移动平均线'] = 0
            else:
                indexMovingAverage.loc[iloop, '长期移动平均线'] = round((indexMovingAverage['累加值'].iloc[iloop]
                                                                  - indexMovingAverage['累加值'].iloc[iloop - longMovingAvgTerm]) / longMovingAvgTerm,2)

            # 增加价格震荡指数, 实际含义为当前指数值和短期移动平均线的偏移值
            if indexMovingAverage['短期移动平均线'].iloc[iloop] == 0:
                indexMovingAverage.loc[iloop, '价格震荡指数'] = 0
            else:
                indexMovingAverage.loc[iloop, '价格震荡指数'] = indexMovingAverage['收盘价'].iloc[iloop] \
                                                          - indexMovingAverage['短期移动平均线'].iloc[iloop]

        self.logger.info('success to compute moving average of %s : processtime %.4f'
                         % (indexName, time.time() - startTime))
        return indexMovingAverage


    def _sql_to_dataframe(self,tableName,sourceTableName,scale):
        if scale == "批量":
            assert ('公司简称' in self.gConfig.keys() and self.gConfig['公司简称'] != NULLSTR) \
                and ('报告时间' in self.gConfig.keys() and self.gConfig['报告时间'] != NULLSTR) \
                and ('报告类型' in self.gConfig.keys() and self.gConfig['报告类型'] != NULLSTR)\
                ,"parameter 公司简称(%s) 报告时间(%s) 报告类型(%s) is not valid parameter"\
                 %(self.gConfig['公司简称'],self.gConfig['报告时间'],self.gConfig['报告类型'])
            #批量处理模式时会进入此分支
            dataFrame = pd.DataFrame(self._get_stock_list(self.gConfig['指数简称']),columns=self.gJsonBase['stockcodeHeader'])
            condition = self._get_condition(dataFrame)
            sql = ''
            sql = sql + '\nselect * '
            sql = sql + '\nfrom %s' % (sourceTableName)
            sql = sql + '\nwhere ' + condition
        order = self.dictTables[tableName]["orderBy"]
        if isinstance(order,list) and len(order) > 0:
            sql = sql + '\norder by ' + ','.join(order)
        dataframe = pd.read_sql(sql, self._get_connect())
        return dataframe


    def _write_to_sqlite3(self, dataFrame:DataFrame,tableName):
        conn = self._get_connect()
        if not self._is_table_exist(conn, tableName):
            # 如果是第一次写入, 则新建表,调用积累的函数写入
            super(InterpreterBase,self)._write_to_sqlite3(dataFrame, tableName)
            return
        sql_df = dataFrame.copy()
        isRecordExist = self._is_record_exist(conn, tableName, sql_df)
        if isRecordExist:
            condition = self._get_condition(sql_df)
            sql = ''
            sql = sql + 'delete from {}'.format(tableName)
            sql = sql + '\nwhere ' + condition
            self._sql_executer(sql)
            self.logger.info("delete from {} where is {}!".format(tableName,sql_df['公司简称'].values[0]))
            sql_df.to_sql(name=tableName, con=conn, if_exists='append', index=False)
            conn.commit()
            self.logger.info("insert into {} where is {}!".format(tableName, sql_df['公司简称'].values[0]))
        else:
            if sql_df.shape[0] > 0:
                sql_df.to_sql(name=tableName,con=conn,if_exists='append',index=False)
                conn.commit()
                self.logger.info("insert into {} where is {}!".format(tableName, sql_df['公司简称'].values[0]))
            else:
                self.logger.error('sql_df is empty!')
        conn.close()


    def _get_class_name(self, gConfig):
        return "analysize"


    def initialize(self,dictParameter = None):
        super(StockAnalysize, self).initialize()
        if dictParameter is not None:
            self.gConfig.update(dictParameter)


def create_object(gConfig):
    stockAnalysize = StockAnalysize(gConfig)
    stockAnalysize.initialize()
    return stockAnalysize