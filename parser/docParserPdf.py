from docBaseClassPdf import *
import pdfplumber

#深度学习模型的基类
class docParserPdf(docBasePdf):
    def __init__(self,gConfig):
        super(docParserPdf,self).__init__(gConfig)
        self.viewIsOn = self.gConfig['viewIsOn'.lower()]
        self.tableKeyword = self.gConfig['tableKeyword'.lower()]
        self.fieldKeyword = self.gConfig['fieldKeyword'.lower()]
        self.excludeKeyword = self.gConfig['excludeKeyword'.lower()]
        if len(self.excludeKeyword) == 1 and self.excludeKeyword[0] == '':
            self.excludeKeyword = list()  #置空

    def parse(self,sourceFile,targetFile):
        sourceFile = os.path.join(self.data_directory,sourceFile)
        targetFile = os.path.join(self.working_directory,targetFile)
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
        findedTableName = ""
        begin_index = int(len(pdf.pages) / 2)
        for i in range(begin_index, len(pdf.pages)):
            if find_table:
                find_pre_table = 1
            else:
                find_pre_table = 0
            find_table = 0
            page = pdf.pages[i]
            # print(page.extract_text())
            data = page.extract_text()
            if len(self.tableKeyword):
                for keyword in self.tableKeyword:
                    if keyword in data:
                        find_table = 1
                        findedTableName = keyword
                        break
                    else:
                        find_table = 0
                        #break
            else:
                find_table = 1

            if find_table or find_pre_table:
                data_list = data.strip().split()
                for j in range(len(data_list)):
                    if len(self.fieldKeyword):
                        for keyword in self.fieldKeyword:
                            if keyword in data_list[j]:
                                find_keyword = 1
                    else:
                        find_keyword = 1

                    if find_keyword:
                        find_keyword = 0
                        print('find %s in page %d'%(findedTableName,i))
                        if len(self.excludeKeyword):
                            for keyword in self.excludeKeyword:
                                if keyword not in data_list[j]:
                                    find_keyword_outside = 1
                                else:
                                    find_keyword_outside = 0
                                    break
                        else:
                            find_keyword_outside = 1

                        if find_keyword_outside:
                            find_keyword_outside = 0
                            name_find.append(data_list[j])
                            value_find.append(data_list[j + 1])
                            page_find.append(i)
                            print("*************find*******************{} value is {}".format(data_list[j],
                                                                                              data_list[j + 1]))
                            print("*************find in page*******************{}".format(i))
                            print("*************find*******************")

        pdf.close()
        start3 = time.time()

        print('****time to open PDF file is {}'.format((start2 - start1)))
        print('****time to processing PDF file is {}'.format((start3 - start2)))

        return name_find, value_find, page_find

def create_model(gConfig):
    parser=docParserPdf(gConfig=gConfig)
    parser.initialize()
    return parser
