#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/6/2020 5:03 PM
# @Author  : wu.hao
# @File    : dataVisualizeation.py
# @Note    : 用于财务数据的可视化

from interpreterAnalysize.interpreterBaseClass import *
from openpyxl import load_workbook,Workbook
import pandas as pd

class ExcelVisualization(InterpreterBase):
    def __init__(self,gConfig):
        super(ExcelVisualization, self).__init__(gConfig)
        self.analysizeresult = os.path.join(self.working_directory,gConfig['analysizeresult'])
        #self.workbook = Workbook()
        self.checkpointIsOn = gConfig['checkpointIsOn'.lower()]

    def _get_class_name(self, gConfig):
        visualization_name = re.findall('(.*)Visualization', self.__class__.__name__).pop().lower()
        #assert dataset_name in gConfig['interpreterlist'], \
        #    'interpreterlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
        #    (gConfig['interpreterlist'], dataset_name, self.__class__.__name__)
        return visualization_name

    def read_and_visualize(self,visualize_file,tableName):
        # 专门用于写文件
        #table = dictTable['table']
        #tableName = dictTable['tableName']
        #workbook = load_workbook(self.targetFile)
        #if workbook.active.title == "Sheet":  # 表明这是一个空工作薄
        #    workbook.remove(workbook['Sheet'])  # 删除空工作薄
        #writer = pd.ExcelWriter(self.targetFile, engine='openpyxl')
        #writer.book = workbook
        visualize_file = os.path.join(self.working_directory,visualize_file)
        assert os.path.exists(visualize_file),"the file %s is not exists,you must create it first!"%visualize_file
        workbook = load_workbook(visualize_file)
        writer = pd.ExcelWriter(visualize_file,engine='openpyxl')
        writer.book = workbook
        sheetNames = workbook.sheetnames
        if tableName not in sheetNames:
            sheetNames = sheetNames + [tableName]
        for sheetName in sheetNames:
            dataframe = self._sql_to_dataframe(sheetName,tableName)

            dataframe = self._process_field_discard(dataframe, tableName)

            self._visualize_to_excel(writer,sheetName,dataframe,tableName)
        workbook.close()
        #writer.close()

    def _sql_to_dataframe(self,sheetName,tableName):
        if sheetName != tableName:
            return
        order = self.dictTables[tableName]["order"]
        sql = ''
        sql = sql + '\nselect * '
        sql = sql + '\nfrom %s'%tableName
        if isinstance(order,list) and len(order) > 0:
            sql = sql + '\norder by ' + ','.join(order)
        dataframe = pd.read_sql(sql, self._get_connect())
        return dataframe

    def _process_field_discard(self, dataFrame, tableName):
        if dataFrame is None:
            return
        fieldDiscard = self.dictTables[tableName]['fieldDiscard']
        if isinstance(fieldDiscard,list) and len(fieldDiscard) > 0:
            dataFrame = dataFrame.drop(labels=fieldDiscard,axis=1)
        return dataFrame

    def _visualize_to_excel(self,writer,sheetName,dataframe:pd.DataFrame,tableName):
        if dataframe is None:
            return
        startrow = self.dictTables[tableName]['startrow']
        dataframe.to_excel(excel_writer=writer, sheet_name=sheetName, index=None,header=True,startrow=startrow)
        #dataframe.to_excel(writer.path,sheet_name=sheetName, index=None,header=False,startrow=startrow)
        writer.save()

    def initialize(self):
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        #suffix = self.analysizeresult.split('.')[-1]
        #assert suffix in self.gConfig['excelSuffix'.lower()], \
        #    'suffix of %s is invalid,it must one of %s' % (self.analysizeresult, self.gConfig['excelSuffix'.lower()])
        #if self.checkpointIsOn == False:
            # 没有启用备份文件时,初始化workbook
            #if os.path.exists(self.analysizeresult):
            #    os.remove(self.analysizeresult)
        #assert os.path.exists(self.analysizeresult),'%s file is not exists,you must create it first!'%self.analysizeresult
        #self.workbook = Workbook()
        #self.writer = pd.ExcelWriter(self.analysizeresult, engine='openpyxl')
        #self.writer.book = self.workbook
        #self.writer.save()  # 生成一个新文件

def create_object(gConfig):
    dataVisualization = ExcelVisualization(gConfig)
    dataVisualization.initialize()
    return dataVisualization