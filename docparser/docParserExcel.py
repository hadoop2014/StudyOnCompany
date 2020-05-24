from docBaseClassExcel import *

class docParserExcel(docBaseExcel):
    def __init__(self,gConfig,writeParser):
        super(docParserExcel,self).__init__(gConfig)
        self.writeparser = writeParser


def create_model(gConfig,writeParser=None):
    parser=docParserExcel(gConfig,writeParser)
    parser.initialize()
    return parser