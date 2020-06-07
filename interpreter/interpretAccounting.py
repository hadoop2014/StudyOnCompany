# -----------------------------------------------------------------------------
# interpretCalc.py
#
# A simple calculator with variables.   This is from O'Reilly's
# "Lex and Yacc", p. 63.
# -----------------------------------------------------------------------------
from interpreter.interpretBaseClass import *


class interpretAccounting(interpretBase):
    def __init__(self,gConfig,docParser):
        super(interpretAccounting,self).__init__(gConfig)
        self.docParser = docParser
        #self.initialize()
        self.interpretDefine()

    def interpretDefine(self):
        tokens = self.tokens

        literals = self.literals

        # Tokens
        #采用动态变量名
        local_name = locals()
        for token in self.tokens:
            local_name['t_'+token] = self.dictTokens[token]
        print(str({key:value for key,value in local_name.items() if key.split('_')[-1] in tokens}).replace("',","'\n"))

        #t_ignore = " \t\n"
        t_ignore = self.ignores

        def t_newline(t):
            r'\n+'
            t.lexer.lineno += t.value.count("\n")


        def t_error(t):
            print("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)

        # Build the lexer
        self.lexer = lex.lex(outputdir=self.working_directory)

        # Parsing rules

        precedence = (
            ('right', 'UMINUS'),
        )

        # dictionary of names
        self.names = {}

        def p_statement_grouphalf(p):
            '''statement : statement ')'
                         | statement '）' '''
            p[0] = p[1]

        def p_statement_statement(p):
            '''statement : statement expression
                         | expression '''
            p[0] = p[1]

        def p_expression_reduce(p):
            '''expression : fetchtable expression
                          | fetchdata expression
                          | skipword '''
            p[0] = p[1]


        def p_fetchtable_search(p):
            '''fetchtable : TABLE optional TIME optional UNIT optional '''
            print("search",p[1],p[3])
            self.names['unit'] = p[5].split(':')[-1].split('：')[-1]
            self.names[p[1]].update({'tableName':p[1],'time':p[3],'unit':self.names['unit'],'currency':self.names['currency']
                                     ,'company':self.names['company'],'tableBegin':True})
            interpretPrefix = '\n'.join([self.names[p[1]]['tableName'], self.names[p[1]]['company'],
                                         self.names[p[1]]['time'], self.names[p[1]]['unit']]) + '\n'
            if self.names[p[1]]['tableEnd'] == False:
                self.docParser._merge_table(self.names[p[1]],interpretPrefix)
            print(self.names[p[1]])

        def p_fetchtable_searchnotime(p):
            '''fetchtable : TABLE optional UNIT optional '''
            print("search",p[1],p[3])
            self.names['unit'] = p[3].split(':')[-1].split('：')[-1]
            self.names[p[1]].update({'tableName':p[1],'unit':self.names['unit'],'currency':self.names['currency']
                                ,'tableBegin':True})
            interpretPrefix = '\n'.join([self.names[p[1]]['tableName'], self.names[p[1]]['unit'],
                                         self.names[p[1]]['currency']]) + '\n'
            if self.names[p[1]]['tableEnd'] == False:
                self.docParser._merge_table(self.names[p[1]],interpretPrefix)
            print(self.names[p[1]])

        def p_fetchtable_skiptime(p):
            '''fetchtable : TABLE optional TIME TIME '''
            #去掉主要会计数据的表头
            print(p[1])

        def p_fetchtable_skipheader(p):
            '''fetchtable : TABLE HEADER '''
            #去掉合并资产负债表项目
            print(p[1])

        def p_fetchtable_skipterm(p):
            '''fetchtable : TABLE term '''
            print(p[1])

        def p_fetchdata_title(p):
            '''fetchdata : COMPANY TIME UNIT '''
            self.names['company'].updata({'company':p[1]})
            self.names['title'].update({'title':p[2]+p[3]})
            print('fetchdata %s %s'%(self.names['company'],self.names['title']))

        def p_skipword_group(p):
            '''skipword : '(' skipword ')'
                        | '（' skipword '）' '''
            p[0] = p[2]

        def p_skipword(p):
            '''skipword : useless skipword
                        | term skipword
                        | useless
                        | term
                        | '(' skipword error '''
            p[0] = p[1]
            #print('skipword',p[0])

        def p_useless_reduce(p):
            '''useless : '(' useless ')'
                       |  '（' useless '）'
                       | '-' useless '''
            p[0] = p[1]

        def p_useless(p):
            '''useless : PUNCTUATION
                       | DISCARD
                       | WEBSITE
                       | EMAIL
                       | NAME
                       | HEADER
                       | COMPANY
                       | TIME
                       | UNIT
                       | CURRENCY
                       | CRITICAL
                       | '-'
                       | '%' '''
            print('useless ',p[1])

        def p_term_group(p):
            '''term : '(' term ')'
                    |  '（' term '）'
                    | '-' term %prec UMINUS '''
            p[0] = -p[2]  #财务报表中()表示负值
            print('uminus',p[0])

        def p_term_percentage(p):
            '''term : NUMERIC '%' '''
            p[0] = round(float(p[1]) * 0.01,4)
            print('percentage',p[0])

        def p_term_numeric(p):
            '''term : NUMERIC '''
            if p[1].find('.') < 0 :
                p[0] = int(p[1].replace(',',''))
            else:
                p[0] = float(p[1].replace(',',''))
            print('value',p[0],p[1])

        def p_optional_optional(p):
            '''optional : DISCARD optional'''

        def p_optional(p):
            '''optional :  empty
                        | CURRENCY
                        | COMPANY '''
            if p.slice[1].type == 'CURRENCY':
                self.names['currency'] = p[1].split(':')[-1].split('：')[-1]
            if p.slice[1].type == 'COMPANY':
                self.names['company'] = p[1]
            p[0] = p[1]
            print('optional',p[0])

        def p_empty(p):
            '''empty : '''
            print('empty')

        def p_error(p):
            if p:
                print("Syntax error at '%s:%s'" % (p.value,p.type))
            else:
                print("Syntax error at EOF")

        # Build the docparser
        self.parser = yacc.yacc(outputdir=self.working_directory)

    def doWork(self,docParser,lexer=None,debug=False,tracking=False):
        #千和财报: 71 - 73 合并资产负债表
        for data in docParser:
            self.parser.parse(docParser._get_text(data).replace(' ',''),lexer=lexer,debug=debug,tracking=tracking)

        '''
        #item = 83,6,120,111,71,149,110,4,38,108,149
        item = 38
        data = docParser._get_item(item)
        text = docParser._get_text(data)
        print(text)
        self.lexer.input(text)
        for token in self.lexer:
            print(token)
        self.parser.parse(text,lexer=self.lexer,debug=True)
        '''
        docParser._close()

    def initialize(self):
        for tableName in self.tablesName:
            self.names.update({tableName:{'tableName':'','time':'','unit':'','currency':''
                                          ,'company':'','table':'','tableBegin':False,'tableEnd':False}})
            self.names.update({'unit':'','currency':'','company':'','title':''})

def create_object(gConfig,docParser=None):
    interpreter=interpretAccounting(gConfig,docParser)
    interpreter.initialize()
    return interpreter

