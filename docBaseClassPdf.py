from docBaseClass import  *
import importlib
import sys
import os.path
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfpage import PDFDocument,PDFPage
from pdfminer.pdfinterp import PDFResourceManager,PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal,LAParams
from pdfminer.pdfinterp import PDFTextState
from PyPDF4 import PdfFileReader,PdfFileWriter

importlib.reload(sys)

#深度学习模型的基类
class docBasePdf(docBase):
    def __init__(self,gConfig):
        super(docBasePdf,self).__init__(gConfig)

    def parse(self,sourceFile,targetFile):
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

    def saveCheckpoint(self):
        pass

    def getSaveFile(self):
        if self.model_savefile == '':
            self.model_savefile = None
            return None
        if self.model_savefile is not None:
            if os.path.exists(self.model_savefile)== False:
               return None
                #文件不存在
        return self.model_savefile

    def removeSaveFile(self):
        if self.model_savefile is not None:
            filename = os.path.join(os.getcwd() , self.model_savefile)
            if os.path.exists(filename):
                os.remove(filename)

    def debug_info(self,info = None):
        if self.debugIsOn == False:
            return
        pass
        return

    def debug(self,layer,name=''):
        pass

    def initialize(self):
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self.clear_logging_directory(self.logging_directory)
        suffix = self.sourceFile.split('.')[-1]
        assert suffix.lower() in self.gConfig['pdfSuffix'.lower()],\
            'suffix of {} is invalid,it must one of {}'.format(self.sourceFile, self.gConfig['pdfSuffix'.lower()])



