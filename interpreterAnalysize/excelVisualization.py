#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/6/2020 5:03 PM
# @Author  : wu.hao
# @File    : dataVisualizeation.py
# @Note    : 用于财务数据的可视化

from interpreterAnalysize.interpreterBaseClass import *
from openpyxl import load_workbook
from openpyxl.styles import Font, colors, Alignment,Border,numbers
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
        #writer.book = workbook
        #sheetNames = workbook.sheetnames

        sheetName = tableName# + str(time.thread_time_ns())
        tableNameTarget = tableName
        #if tableName not in sheetNames:
        #    sheetNames = sheetNames + [tableName]
        #for sheetName in sheetNames:
        #    dataframe = self._sql_to_dataframe(sheetName,tableName)

        #    dataframe = self._process_field_discard(dataframe, tableName)

        #    self._visualize_to_excel(writer,sheetName,dataframe,tableName)
        dataframe = self._sql_to_dataframe(sheetName, tableName)

        dataframe = self._process_field_discard(dataframe, tableName)

        self._visualize_to_excel(writer,sheetName,dataframe,tableName)

        self._adjust_style_excel(workbook,sheetName,tableName)
        workbook.save(visualize_file)
        workbook.close()
        #writer.close()

    def _adjust_style_excel(self,workbook,sheetName,tableName):
        font_settings = self.dictTables[tableName]['font_settings']
        startrow = self.dictTables[tableName]['startrow']
        font_common = Font(name=font_settings['name'], size=font_settings['size'], italic=font_settings['italic']
                    ,color=colors.COLOR_INDEX[font_settings['color_index_common']], bold=font_settings['bold'], underline=None)
        font_emphasize = Font(name=font_settings['name'], size=font_settings['size'], italic=font_settings['italic']
                           , color=colors.COLOR_INDEX[font_settings['color_index_emphasize']], bold=font_settings['bold'],
                           underline=None)
        alignment = Alignment(horizontal='right', vertical='center', wrap_text=True)
        border = Border()
        sheet = workbook[sheetName]
        maxrow = self._get_maxrow(sheet.max_row,tableName)

        for i,cell in enumerate(sheet[startrow + 1]):
            cell.font = font_common
            if self._is_cell_emphasize(cell,tableName):
                cell.font = font_emphasize
            if self._is_cell_pecentage(cell,tableName):
                for index in range(maxrow):
                    sheet.cell(row = index + 1,column = i + 1).number_format = numbers.FORMAT_PERCENTAGE
            cell.alignment = alignment
            cell.border = border #去掉边框

        #blue_fill = PatternFill(start_color='FF0000FF', end_color='FF0000FF', fill_type='solid')

        #dxf = DifferentialStyle(fill=blue_fill, numFmt=NumFmt(10, '0.00%'))
        #rule = Rule('expression', formula=['E3 > 3'], dxf=dxf)
        #ws.conditional_formatting.add('E3', rule)

    def _sql_to_dataframe(self,sheetName,tableName):
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

    def _get_maxrow(self,current_maxrow,tableName):
        maxrow = self.dictTables[tableName]['maxrow']
        return max(maxrow,current_maxrow)

    def _is_cell_pecentage(self,cell,tableName):
        isCellPecentage = False
        percentage_exclude = self.dictTables[tableName]['percentage_exclude']
        if self._is_cell_emphasize(cell,tableName):
            pattern_percentage_exclude = '|'.join(percentage_exclude)
            isCellPecentage = not self._is_matched(pattern_percentage_exclude,cell.value)
        return isCellPecentage

    def _is_cell_emphasize(self,cell,tableName):
        pattern_emphasize = self.dictTables[tableName]['pattern_emphasize']
        isCellEmphasize = self._is_matched(pattern_emphasize,cell.value)
        return isCellEmphasize

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