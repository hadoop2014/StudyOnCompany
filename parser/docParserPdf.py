from docBaseClassPdf import *


#深度学习模型的基类
class docParserPdf(docBasePdf):
    def __init__(self,gConfig):
        super(docParserPdf,self).__init__(gConfig)
        self.viewIsOn = self.gConfig['viewIsOn'.lower()]

def create_model(gConfig):
    parser=docParserPdf(gConfig=gConfig)
    parser.initialize()
    return parser
