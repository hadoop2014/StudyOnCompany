from docparser.docParserBaseClass import *
import pdfplumber
import pandas as pd
from openpyxl import Workbook

class docParserPdf(docParserBase):
    def __init__(self,gConfig,writeParser):
        super(docParserPdf,self).__init__(gConfig)
        self.writeParser = writeParser
        self.workbook = Workbook()

    def parse(self,sourceFile,targetFile):
        sourceFile = os.path.join(self.data_directory,sourceFile)
        #targetFile = os.path.join(self.working_directory,targetFile)
        #self.writeParser.initialize(targetFile)
        dictKeyword = self.dictKeyword
        start1 = time.time()

        pdf = pdfplumber.open(sourceFile, password='')
        start2 = time.time()

        find_table = 0
        find_pre_table = 0
        find_keyword = 0
        find_keyword_outside = 0
        name_find = []
        value_find = []
        page_find = []
        # for page in pdf.pages:
        # print(page.extract_text())
        findedTableKeyword = ""
        #begin_index = int(len(pdf.pages) / 2)
        begin_index = 1
        for page_no in range(begin_index, len(pdf.pages)):
            if find_table:
                find_pre_table = 1
            else:
                find_pre_table = 0
            find_table = 0
            page = pdf.pages[page_no]
            # print(page.extract_text())
            data = page.extract_text()
            if len(self.tableKeyword):
                for keyword in self.tableKeyword:
                    if keyword in data:
                        find_table = 1
                        findedTableKeyword = keyword
                        break
                    else:
                        find_table = 0
                        #break
            else:
                find_table = 1

            if find_table or find_pre_table:
                tables = page.extract_tables() #解析所有的表格
                for index,table in enumerate(tables):
                    dataframe = pd.DataFrame(table[1:], columns=table[0],index=None)  # 以第一行为列变量
                    #tb.to_excel(targetFile,index=False)  #不显示索引
                    self.writeParser.writeToExcel(dataframe,sheetName=findedTableKeyword+str(index))

                data_list = data.strip().split()
                fieldKeyword = dictKeyword[findedTableKeyword]['fieldKeyword']
                for row_no in range(len(data_list)):
                    if len(fieldKeyword):
                        for keyword in fieldKeyword:
                            if keyword in data_list[row_no]:
                                find_keyword = 1
                    else:
                        find_keyword = 1

                    if find_keyword:
                        find_keyword = 0
                        print('find %s in page %d'%(findedTableKeyword,page_no))
                        excludeKeyword = dictKeyword[findedTableKeyword]['excludeKeyword']
                        if len(excludeKeyword):
                            for keyword in excludeKeyword:
                                if keyword not in data_list[row_no]:
                                    find_keyword_outside = 1
                                else:
                                    find_keyword_outside = 0
                                    break
                        else:
                            find_keyword_outside = 1

                        if find_keyword_outside:
                            find_keyword_outside = 0
                            name_find.append(data_list[row_no])
                            value_find.append(data_list[row_no + 1])
                            page_find.append(page_no)
                            print("*************find*******************{} value is {}\n".format(data_list[row_no],
                                                                                              data_list[row_no + 1]))
                            #print("*************find in page*******************{}".format(page_no))
                            #print("*************find*******************")

        pdf.close()
        start3 = time.time()

        print('****time to open PDF file is {}'.format((start2 - start1)))
        print('****time to processing PDF file is {}'.format((start3 - start2)))

        return name_find, value_find, page_find

    def parsePdfminer(self,sourceFile,targetFile):
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfpage import PDFDocument, PDFPage
        from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer.converter import PDFPageAggregator
        from pdfminer.layout import LTTextBoxHorizontal, LAParams
        from pdfminer.pdfinterp import PDFTextState
        '''解析PDF文本，并保存到TXT文件中'''
        sourceFile = os.path.join(self.data_directory,sourceFile)
        targetFile = os.path.join(self.working_directory,targetFile)
        targetFile = '.'.join([*targetFile.split('.')[:-1],'txt'])
        if os.path.exists(targetFile):
            os.remove(targetFile)
        fp = open(sourceFile, 'rb')
        # 用文件对象创建一个PDF文档分析器
        parser = PDFParser(fp)
        # 创建一个PDF文档
        doc = PDFDocument(parser)
        # 连接分析器，与文档对象
        parser.set_document(doc)
        #doc.set_parser(parser)

        # 提供初始化密码，如果没有密码，就创建一个空的字符串
        #doc.initialize()

        # 检测文档是否提供txt转换，不提供就忽略
        if not doc.is_extractable:
            raise PDFTextState.PDFTextExtractionNotAllowed
        else:
            # 创建PDF，资源管理器，来共享资源
            rsrcmgr = PDFResourceManager()
            # 创建一个PDF设备对象
            laparams = LAParams()
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            # 创建一个PDF解释其对象
            interpreter = PDFPageInterpreter(rsrcmgr, device)

            # 循环遍历列表，每次处理一个page内容
            # doc.get_pages() 获取page列表
            #for page in doc.get_pages():
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
                # 接受该页面的LTPage对象
                layout = device.get_result()
                # 这里layout是一个LTPage对象 里面存放着 这个page解析出的各种对象
                # 一般包括LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等等
                # 想要获取文本就获得对象的text属性，
                with open(targetFile, 'a') as f:
                    for x in layout:
                        if (isinstance(x, LTTextBoxHorizontal)):
                            results = x.get_text()
                            print(results)
                            f.write(results + "\n")

            outlines = doc.get_outlines()
            for (level, title, dest, a, se) in outlines:
                print(level, title)

    def initialize(self):
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self.clear_logging_directory(self.logging_directory)
        suffix = self.sourceFile.split('.')[-1]
        assert suffix.lower() in self.gConfig['pdfSuffix'.lower()], \
            'suffix of {} is invalid,it must one of {}'.format(self.sourceFile, self.gConfig['pdfSuffix'.lower()])

def create_object(gConfig, writeParser=None):
    parser=docParserPdf(gConfig,writeParser)
    parser.initialize()
    return parser
