from docBaseClass import  *

#深度学习模型的基类
class docBaseSql(docBase):
    def __init__(self,gConfig):
        super(docBaseSql,self).__init__(gConfig)

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


