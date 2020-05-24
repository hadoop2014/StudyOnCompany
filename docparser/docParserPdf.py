from docBaseClassPdf import *
import pdfplumber
import pandas as pd

#深度学习模型的基类
class docParserPdf(docBasePdf):
    def __init__(self,gConfig):
        super(docParserPdf,self).__init__(gConfig)

    def parse(self,sourceFile,targetFile):
        sourceFile = os.path.join(self.data_directory,sourceFile)
        targetFile = os.path.join(self.working_directory,targetFile)
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
        begin_index = int(len(pdf.pages) / 2)
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
                for table in tables:
                    tb = pd.DataFrame(table[1:], columns=table[0],index=None)  # 以第一行为列变量
                    tb.to_excel(targetFile,index=False)  #不显示索引
                    #tb.to_csv(targetFile)
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

def create_model(gConfig):
    parser=docParserPdf(gConfig=gConfig)
    parser.initialize()
    return parser
