#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/6/2020 5:03 PM
# @Author  : wu.hao
# @File    : dataVisualizeation.py
# @Note    : 用于财务数据的可视化

from interpreterAnalysize.interpreterBaseClass import *
from openpyxl import load_workbook,Workbook
from openpyxl.styles import Font, colors, Alignment,Border,numbers,PatternFill
from openpyxl.utils import get_column_letter,get_column_interval
from openpyxl.formatting import Rule
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule, FormulaRule
from openpyxl.styles.differential import DifferentialStyle
import pandas as pd
from functools import partial


class CheckpointModelVisual(CheckpointModelBase) :

    '''
    explain: 该类用于保持模型数据所存放的文件的管理, 文件个数受max_keep_models控制,即一个model对应一个输出文件, 文件类型为xlsx
    '''
    def _init_modelfile(self, visualize_file):
        '''
        explain: 如果文件不存在,则创建它; 如果文件存在,尝试打开它,如果打不开,则删除该文件,重新创建.
        '''
        workbook = Workbook()
        if not os.path.exists(visualize_file):
            # 生成一个新文件
            workbook.save(visualize_file)
        try:
            workbook = load_workbook(visualize_file)
        except Exception as e:
            # 删除文件重新执行一次
            print(e)
            self.logger.info('failed to load workbook,remove it and try load it again from file %s, !' % visualize_file)
            if os.path.exists(visualize_file):
                os.remove(visualize_file)
                workbook.save(visualize_file)
        finally:
            workbook.close()


    @classmethod
    def copy_file_for_reader(cls,func):
        '''
        explain: 专门用于装饰read_and_visualize函数
            1) 在该函数运行完后, 将结果excel拷贝一份到目标modelfile_basic中,这样每次可以用wps打开一个固定的excel.
            2) 因为cls.modelfile_basic是类变量,在多进程下会被其他进程覆盖,因此在多进程可能存在问题
        '''
        @functools.wraps(func)
        def wrap(self,visualize_file, *args):
            func(self, visualize_file, *args)
            if visualize_file != cls.modelfile_basic and cls.modelfile_basic != NULLSTR:
                if os.path.exists(visualize_file):
                    shutil.copyfile(visualize_file, cls.modelfile_basic)
        return wrap


class ExcelVisualization(InterpreterBase):
    def __init__(self,gConfig):
        super(ExcelVisualization, self).__init__(gConfig)
        # 用于创建visual对象的偏函数,该对象增加checkpoint属性, 利用max_keep_models参数控制文件个数
        self.create_visual =  partial(self.create_space
                                      , CheckpointModelVisual
                                      , copy_file = True
                                      , max_keep_models=self.gConfig['max_keep_models'])

    @CheckpointModelVisual.copy_file_for_reader
    def read_and_visualize(self,visualize_file, tableName,scale):
        '''
        explian: 将结果数据写入excel文件并按照指定格式呈现
        '''
        workbook,writer = self.get_workbook_and_writer(visualize_file)
        for reportType in self.gConfig['报告类型']:
            tablePrefix = self.standard._get_tableprefix_by_report_type(reportType)
            sheetName = tablePrefix + tableName# + str(time.thread_time_ns())
            if sheetName in workbook.sheetnames:
                # 先清空旧的工作薄
                workbook.remove(workbook[sheetName])

            dataframe = self._sql_to_dataframe(tableName,sourceTableName=sheetName,scale=scale)

            dataframe = self._process_field_discard(dataframe, tableName)

            self._visualize_to_excel(writer,sheetName,dataframe,tableName)

            self._adjust_style_excel(workbook,sheetName,tableName)

            if len(workbook._sheets) > 1:
                targetSheet = workbook[sheetName]
                workbook._sheets[1:] = workbook._sheets[:-1]
                workbook._sheets[0] = targetSheet
            self.logger.info('%s 展示 %s 成功!' % (scale, sheetName))

        workbook.save(visualize_file)
        workbook.close()


    def get_workbook_and_writer(self,visualize_file):
        '''
        explain: 根据文件名获取workbook和writer
        '''
        workbook =load_workbook(visualize_file)
        writer = pd.ExcelWriter(visualize_file, engine='openpyxl')
        writer.book = workbook
        if workbook.active.title == "Sheet":  # 表明这是一个空工作薄
            workbook.remove(workbook['Sheet'])  # 删除空工作薄
        return workbook, writer


    def _adjust_style_excel(self,workbook,sheetName,tableName):
        sheet = workbook[sheetName]

        self._set_height_and_width(sheet,tableName)

        self._set_freeze_pans(sheet,tableName)

        self._set_font_and_style(sheet,tableName)

        self._set_conditional_formatting(sheet,tableName)

        #blue_fill = PatternFill(start_color='FF0000FF', end_color='FF0000FF', fill_type='solid')
        #dxf = DifferentialStyle(fill=blue_fill, numFmt=NumFmt(10, '0.00%'))
        #rule = Rule('expression', formula=['E3 > 3'], dxf=dxf)
        #ws.conditional_formatting.add('E3', rule)


    def _sql_to_dataframe(self,tableName,sourceTableName,scale):
        if scale == "批量":
            assert ('公司简称' in self.gConfig.keys() and self.gConfig['公司简称'] != NULLSTR) \
                and ('报告时间' in self.gConfig.keys() and self.gConfig['报告时间'] != NULLSTR) \
                and ('报告类型' in self.gConfig.keys() and self.gConfig['报告类型'] != NULLSTR)\
                ,"parameter 公司简称(%s) 报告时间(%s) 报告类型(%s) is not valid parameter"\
                 %(self.gConfig['公司简称'],self.gConfig['报告时间'],self.gConfig['报告类型'])
            #批量处理模式时会进入此分支
            sql = ''
            sql = sql + '\nselect * '
            sql = sql + '\nfrom %s' % (sourceTableName)
            sql = sql + '\nwhere (' + ' or '.join(['公司简称 =' + '\'' +  company + '\''   for company in self.gConfig['公司简称']]) + ')'
            sql = sql + '    and (' + ' or '.join(['报告时间 =' + '\'' + reporttime + '\'' for reporttime in self.gConfig['报告时间']]) + ')'
            sql = sql + '    and (' + ' or '.join(['报告类型 =' + '\'' + reportype + '\'' for reportype in self.gConfig['报告类型']]) + ')'
        elif scale == "定量":
            assert ('公司组合' in self.gConfig.keys() and self.gConfig['公司组合']) \
                   and ('报告时间' in self.gConfig.keys() and self.gConfig['报告时间']) \
                   and ('报告类型' in self.gConfig.keys() and self.gConfig['报告类型']) \
                , "parameter 公司组合(%s) 报告时间(%s) 报告类型(%s) is not valid parameter" \
                  % (self.gConfig['公司组合'], self.gConfig['报告时间'], self.gConfig['报告类型'])
            # 批量处理模式时会进入此分支
            sql = ''
            sql = sql + '\nselect * '
            sql = sql + '\nfrom %s' % (sourceTableName)
            sql = sql + '\nwhere (' + ' or '.join(
                ['公司简称 =' + '\'' + company + '\'' for company in self.gConfig['公司组合']]) + ')'
            sql = sql + '    and (' + ' or '.join(
                ['报告时间 =' + '\'' + reporttime + '\'' for reporttime in self.gConfig['报告时间']]) + ')'
            sql = sql + '    and (' + ' or '.join(
                ['报告类型 =' + '\'' + reportype + '\'' for reportype in self.gConfig['报告类型']]) + ')'
        else:
            sql = ''
            sql = sql + '\nselect * '
            sql = sql + '\nfrom %s' % (sourceTableName)
        order = self.dictTables[tableName]["orderBy"]
        if isinstance(order,list) and len(order) > 0:
            sql = sql + '\norder by ' + ','.join(order)
        dataframe = pd.read_sql(sql, self.database._get_connect())
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
        writer.save()


    def _set_conditional_formatting(self,sheet,tableName):
        req = ':([a-zA-Z]+)'
        match = re.findall(req, sheet.dimensions)
        assert isinstance(match, list) and len(match) > 0, 'sheet.dimensions(%s) is invalid!' % sheet.dimensions
        startrow = self.dictTables[tableName]['startrow']
        maxrow = self._get_maxrow(sheet.max_row,tableName)

        conditional_formatting = self.dictTables[tableName]['conditional_formatting']
        conditional_field = conditional_formatting.keys()
        pattern_field = '|'.join(conditional_field)
        weight = match[0]
        letter_list = get_column_interval('A', weight)
        for col_letter in letter_list:
            col = sheet[col_letter]
            field_name = col[startrow].value
            if utile.is_matched(pattern_field, field_name):
                operator = conditional_formatting[field_name]['operator']
                if operator != NULLSTR:
                    threhold = conditional_formatting[field_name]['threshold']
                    col[0].value = threhold
                    color_index = conditional_formatting[field_name]['color_index']
                    font = Font(color=colors.COLOR_INDEX[color_index])
                    rule = CellIsRule(operator=operator, formula=[str(threhold)], stopIfTrue=False, font = font)
                    rule_applys = col_letter + str(startrow + 2) + ":" + col_letter + str(maxrow)
                    sheet.conditional_formatting.add(rule_applys, rule)
        return


    def _set_font_and_style(self,sheet, tableName):
        font_settings = self.dictTables[tableName]['font_settings']
        startrow = self.dictTables[tableName]['startrow']
        font_common = Font(name=font_settings['name'], size=font_settings['size'], italic=font_settings['italic']
                    ,color=colors.COLOR_INDEX[font_settings['color_index_common']], bold=font_settings['bold'], underline=None)
        font_emphasize = Font(name=font_settings['name'], size=font_settings['size'], italic=font_settings['italic']
                           , color=colors.COLOR_INDEX[font_settings['color_index_emphasize']], bold=font_settings['bold'],
                           underline=None)
        alignment_header = Alignment(horizontal='left', vertical='center', wrap_text=True)
        alignment_field = Alignment(horizontal='right', vertical='center', wrap_text=False
                                    ,justifyLastLine=True)
        border = Border()
        maxrow = self._get_maxrow(sheet.max_row,tableName)
        freezecol = self.dictTables[tableName]['freezecol']
        builtin_formats = self.dictTables[tableName]['builtin_formats']

        for i, cell in enumerate(sheet[startrow + 1]):
            cell.font = font_common
            cell.alignment = alignment_header
            cell.border = border  # 去掉边框
            if self._is_cell_emphasize(cell, tableName):
                cell.font = font_emphasize
            for index in range(maxrow):
                if self._is_cell_pecentage(cell, tableName):
                    sheet.cell(row=index + 1, column = i + 1).number_format = numbers.FORMAT_PERCENTAGE_00
                elif i + 1 >= freezecol:
                    sheet.cell(row=index + 1, column = i + 1).number_format = numbers.BUILTIN_FORMATS[builtin_formats]
                if index > startrow:
                    sheet.cell(row=index + 1, column = i + 1).alignment = alignment_field


    def _set_freeze_pans(self,sheet,tableName):
        startrow = self.dictTables[tableName]['startrow']
        freezecol = self.dictTables[tableName]['freezecol']
        sheet.freeze_panes = get_column_letter(freezecol) + str(startrow + 2)


    def _set_height_and_width(self,sheet,tableName):
        req = ':([a-zA-Z]+)'
        match = re.findall(req, sheet.dimensions)
        assert isinstance(match,list) and len(match) > 0,'sheet.dimensions(%s) is invalid!'%sheet.dimensions
        #设置标题头的行高
        startrow = self.dictTables[tableName]['startrow']
        maxheight = self.dictTables[tableName]['maxheight']
        sheet.row_dimensions[startrow].height = maxheight
        #设置列宽
        weight = match[0]
        letter_list = get_column_interval('A',weight)
        for col_letter in letter_list:
            widthAdjust = self._get_col_max_width(sheet,col_letter,tableName)
            sheet.column_dimensions[col_letter].width = widthAdjust


    def _get_col_max_width(self,sheet,col_letter,tableName):
        #_get_col_max_width放在_adjust_style_excel的前面跑,这样设置才有效
        maxrow = sheet.max_row
        maxwidth = self.dictTables[tableName]['maxwidth']
        minwidth = self.dictTables[tableName]['minwidth']
        marginwidth = self.dictTables[tableName]['marginwidth']
        startrow = self.dictTables[tableName]['startrow']
        widthAdjust = 0
        col = sheet[col_letter]
        #跳过标题行
        for i in range(startrow + 1,maxrow):
            if len(str(col[i].value)) > widthAdjust:
                widthAdjust = len(str(col[i].value)) + marginwidth
        widthAdjust = min(widthAdjust,maxwidth)
        widthAdjust = max(widthAdjust,minwidth)
        return widthAdjust


    def _column_char_to_integer(self,name):
        assert isinstance(name,str),'the name(%s) of column must be str'%name
        index = 0
        for char in list(name.upper()):
            index = index * 26 + ord(char) - ord('A') + 1
        return index


    def _column_integer_to_char(self,index):
        assert isinstance(index,int),'the index(%s) of column must be integer'%index
        name = NULLSTR
        base = ord('Z') - ord('A') + 1
        while index >= base:
            name = name + chr(index // base + 64).upper()
            index = index - (index // base) * base
        name = name + chr(index % base + 65).upper()
        return name


    def _get_maxrow(self,current_maxrow,tableName):
        maxrow = self.dictTables[tableName]['maxrow']
        return max(maxrow,current_maxrow)


    def _is_cell_pecentage(self,cell,tableName):
        percentage_field = [key for key,value in self.dictTables[tableName]['conditional_formatting'].items()
                            if value['value_format'] == 'percentage']
        pattern_percentage_field = '|'.join(percentage_field)
        isCellPecentage = utile.is_matched(pattern_percentage_field, cell.value)
        return isCellPecentage


    def _is_cell_emphasize(self,cell,tableName):
        pattern_emphasize = self.dictTables[tableName]['pattern_emphasize']
        isCellEmphasize = utile.is_matched(pattern_emphasize, cell.value)
        return isCellEmphasize


    def initialize(self,dictParameter = None):
        if dictParameter is not None:
            self.gConfig.update(dictParameter)


def create_object(gConfig):
    dataVisualization = ExcelVisualization(gConfig)
    dataVisualization.initialize()
    return dataVisualization