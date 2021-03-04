#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 6/13/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterAnalysize.py
# @Note    : 用于财务数据分析

from interpreterAnalysize.interpreterBaseClass import *
from interpreterAssemble import InterpreterAssemble
import matplotlib.pyplot as plt
from time import sleep
from ply import lex,yacc


class InterpreterAnalysize(InterpreterBase):
    def __init__(self,gConfig,memberModuleDict):
        super(InterpreterAnalysize, self).__init__(gConfig)
        self.excelVisualization = memberModuleDict['excelVisualization']
        self.stockAnalysize = memberModuleDict['stockAnalysize']
        self.modelLenetH = memberModuleDict['modelLenetH']
        self.modelLenetM = memberModuleDict['modelLenetM']
        self.modelRnnM = memberModuleDict['modelRnnM']
        self.modelRnnH = memberModuleDict['modelRnnH']
        self.modelRnnregressionH = memberModuleDict['modelRnnregressionH']
        self.modelRnndetectionH = memberModuleDict['modelRnndetectionH']
        self.modelSets = dict({'lenetPytorch': self.modelLenetH})
        self.modelSets.update({'lenetMxnet': self.modelLenetM})
        self.modelSets.update({'rnnMxnet':self.modelRnnM})
        self.modelSets.update({'rnnPytorch':self.modelRnnH})
        self.modelSets.update({'rnnregressionPytorch':self.modelRnnregressionH})
        self.modelSets.update({'rnndetectionPytorch':self.modelRnndetectionH})
        self.interpretDefine()


    def interpretDefine(self):
        tokens = self.tokens
        literals = self.literals
        # Tokens
        #采用动态变量名
        local_name = locals()
        for token in self.tokens:
            local_name['t_'+token] = self.dictTokens[token]
        self.logger.info('\n'+str({key:value for key,value in local_name.items() if key.split('_')[-1] in tokens}).replace("',","'\n"))

        #t_ignore = " \t\n"
        t_ignore = self.ignores
        t_ignore_COMMENT = r'#.*'


        def t_newline(t):
            r'\n+'
            t.lexer.lineno += t.value.count("\n")


        def t_error(t):
            print("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)

        # Build the lexer
        self.lexer = lex.lex(outputdir=self.working_directory,reflags=int(re.MULTILINE))

        # dictionary of names_global
        self.names = {}


        def p_statement_expression(p):
            '''statement : statement expression
                         | expression'''
            pass


        def p_expression_manipulate_table(p):
            '''expression : MANIPULATE TABLE'''
            manipulate = p[1]
            tableName = p[2]
            self._process_manipulate_table(tableName, manipulate)


        def p_expression_batch_analysize(p):
            '''expression : SCALE EXECUTE ANALYSIZE'''
            scale = p[1]
            analysize = p[3]
            self._process_stock_analysize(scale,analysize)


        def p_expression_visualize_table(p):
            '''expression : SCALE VISUALIZE TABLE'''
            tableName = p[3]
            scale = p[1]
            self._process_visualize_table(tableName,scale)


        def p_expression_handle_model(p):
            '''expression : HANDLE MODEL'''
            handle = p[1]
            modelName = p[2]
            self._process_handle_model(modelName, handle)


        def p_error(p):
            if p:
                print("Syntax error at '%s:%s'" % (p.value,p.type))
                self.logger.error("Syntax error at '%s:%s'" % (p.value, p.type))
            else:
                print("Syntax error at EOF page")
                self.logger.error("Syntax error at EOF page")

        # Build the docparser
        self.parser = yacc.yacc(outputdir=self.working_directory)


    def doWork(self,commond,lexer=None,debug=False,tracking=False):
        text = commond
        self.parser.parse(text,lexer=self.lexer,debug=debug,tracking=tracking)


    def _process_stock_analysize(self,scale, analysize):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_manipulate_table!')
            return
        # 获取分析结果所保存的表
        tableName = self.gJsonInterpreter['analysizeTable'][analysize]
        assert tableName in self.tableNames, 'tableName(%s) is invalid ,it must be in %s' % self.tableNames
        if scale != '批量':
            self.logger.warning('the scale %s is not support,now only support scale \'批量\'' % p[1])
        if analysize == '指数趋势分析':
            self.stockAnalysize.initialize(self.gConfig)
            self.stockAnalysize.stock_index_trend_analysize(tableName,scale)
        else:
            self.logger.warning('the analysize %s is not supporte, now only support analysize \' 指数趋势分析 \'')


    def _process_manipulate_table(self, tableName,manipulate):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_manipulate_table!')
            return
        sqlFilename = self._manipulate_transfer(manipulate)
        for reportType in self.gConfig['报告类型']:
            sql_file = self.dictTables[tableName][sqlFilename]
            tablePrefix = self._get_tableprefix_by_report_type(reportType)
            sql_file = os.path.join(self.program_directory,tablePrefix,sql_file)
            if not os.path.exists(sql_file):
                self.logger.error('%s script is not exist,you must create it first :%s!'% (manipulate, sql_file))
                continue
            manipulate_sql = self._get_file_context(sql_file)
            isSuccess = self._sql_executer_script(manipulate_sql)
            assert isSuccess,"failed to execute sql"


    def _process_visualize_table(self,tableName,scale):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_visualize_table!')
            return
        visualize_file = self.dictTables[tableName]['visualize']
        if visualize_file == NULLSTR:
            self.logger.warning('the visualize of table %s is NULL,it can not be visualized!'%tableName)
            return
        self.excelVisualization.initialize(self.gConfig)
        self.excelVisualization.read_and_visualize(visualize_file, tableName, scale)


    def _process_visualize_table_batch(self,tableName):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_visualize_table!')
            return
        visualize_file = self.dictTables[tableName]['visualize']
        if visualize_file == NULLSTR:
            self.logger.warning('the visualize of table %s is NULL,it can not be visualized!'%tableName)
            return
        self.excelVisualization.read_and_visualize(visualize_file, tableName)


    def _process_handle_model(self, modelName,handle):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_visualize_table!')
            return
        assert modelName in self.models,'model(%s) must be in model list:%s'%(modelName, self.models)
        dictModel = self.dictModels[modelName]
        modelName = dictModel['model']
        dataset = dictModel['dataset']
        framework = dictModel['framework']
        if handle == '应用':
            dictModel.update({'mode': "apply"}) # 如果是应用模型来做预测,则把 mode 设置为 apply
        # 当前只支持针对年度报告的预测功能,针对季报的预测功能待开发
        dictModel.update({'报告类型': ['年度报告']})
        gConfig = self.gConfig
        gConfig.update(dictModel)
        getdataClass = InterpreterAssemble().get_data_class(gConfig,dataset)
        dictModel.update({'getdataClass':getdataClass})
        modelModule = self.modelSets[modelName + framework.title()]
        modelModule.initialize(dictModel)
        self.handleStart(modelModule, modelModule, gConfig, handle)
        #self.logger.info("Reatch the interpreterAnalysize just for debug : train %s" % modelName)


    def handleStart(self, model, model_eval, gConfig,handle):
        getdataClass = gConfig['getdataClass']
        framework = gConfig['framework']
        modelName = gConfig['model']
        dataset = gConfig['dataset']
        mode = gConfig['mode']
        if gConfig['unittestIsOn'.lower()] == True:
            num_epochs = 1
        else:
            num_epochs = gConfig['train_num_epoch']
        start_time = time.time()

        if handle == '训练':
            self.logger.info("\n\n(%s %s %s %s) is starting, use optimizer %s,ctx=%s,initializer=%s,check_point=%s,"
                  "activation=%s...............\n\n"
                  % (modelName, framework, dataset, mode, gConfig['optimizer'], gConfig['ctx'], gConfig['initializer'],
                     gConfig['ckpt_used'], gConfig['activation']))
            losses_train, acces_train, losses_valid, acces_valid, losses_test, acces_test = \
                model.train(model_eval, getdataClass, gConfig, num_epochs)
            getdataClass.endProcess()
            self.logger.info('training %s end, time used %.4f' % (modelName, (time.time() - start_time)))
            self.plotLossAcc(losses_train, acces_train, losses_valid, acces_valid, losses_test, acces_test, gConfig, modelName)
        elif handle == '应用':
            self.logger.info("Starting apply %s at (%s %s) to predict ..................................... "
                             % (modelName, framework, dataset))
            model.apply_model(model_eval.net)
            self.logger.info('apply model %s end, time used %.4f\n\n' % (modelName, (time.time() - start_time)))


    def plotLossAcc(self,losses_train, acces_train, losses_valid, acces_valid, losses_test, acces_test, gConfig, taskName):
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(np.reshape(losses_train, [-1]), 'g', label='train loss')
        if losses_test[0] is not None:
            ax1.plot(np.reshape(losses_test, [-1]), 'r-', label='test loss')
        if losses_valid[0] is not None:
            ax1.plot(np.reshape(losses_valid, [-1]), 'r-', label='valid loss')

        ax1.legend()
        ax1.set_ylabel('loss')
        plt.title(taskName, loc='center')

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(np.reshape(acces_train, [-1]), 'g', label='train accuracy')
        if acces_test[0] is not None:
            ax2.plot(np.reshape(acces_test, [-1]), 'r-', label='test accuracy')
        if acces_valid[0] is not None:
            ax2.plot(np.reshape(acces_valid, [-1]), 'r-', label='valid accuracy')
        ax2.set_ylabel('accuracy')
        ax2.legend()
        ax2.set_xlabel(format('epochs (per %d)' % gConfig['epoch_per_print']))
        # thread = Thread(target=closeplt, args=(gConfig['pltsleeptime'],))
        # thread.start()
        if gConfig['unittestIsOn'.lower()] == False:
            # 在unittest模式下，不需要进行绘图，否则会阻塞后续程序运行
            plt.show()


    def _manipulate_transfer(self,manipulate):
        manipulate_transfer = {
            "创建": "create",
            "更新": "update"
        }
        sqlFilename = manipulate_transfer[manipulate]
        return sqlFilename


    def closeplt(self,time):
        sleep(time)
        plt.close()


    def initialize(self,dictParameter=None):
        if dictParameter is not None:
            self.gConfig.update(dictParameter)


def create_object(gConfig,memberModuleDict):
    interpreter=InterpreterAnalysize(gConfig, memberModuleDict)
    interpreter.initialize()
    return interpreter