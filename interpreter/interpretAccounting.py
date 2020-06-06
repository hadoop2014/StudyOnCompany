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
        self._page = None
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
            #('left', '+', '-'),
            #('left', '*', '/'),
            ('right', 'UMINUS'),
        )

        # dictionary of names
        names = {}

        def p_statement_grouphalf(p):
            '''statement : statement ')'
                         | statement '）' '''
            p[0] = p[1]

        def p_statement_statement(p):
            '''statement : statement expression
                         | expression '''
            p[0] = p[1]

        #def p_expression_group(p):
        #    '''expression : '(' expression ')'
        #                  | '（' expression '）' '''
        #    p[0] = p[2]

        def p_expression_reduce(p):
            '''expression : fetchtable expression
                          | skipword '''
            p[0] = p[1]

        #def p_expression_currency(p):
        #    '''expression : CURRENCY '''
        #    names['currency'] = p[1].split(':')[-1].split('：')[-1]
        #    p[0] = p[1]
        #    print(p[0])

        #def p_expression_unit(p):
        #    '''expression : UNIT '''
        #    names['unit'] = p[1].split(':')[-1].split('：')[-1]
        #    p[0] = p[1]
        #    print(p[0])

        #def p_expression_time(p):
        #    '''expression : TIME '''
        #    names['time'] = p[1]
        #    p[0] = p[1]
        #    print(p[0])

        def p_fetchtable_search(p):
            '''fetchtable : TABLE optional TIME optional UNIT optional '''
            print("search",p[1],p[3])
            names['unit'] = p[5].split(':')[-1].split('：')[-1]
            names[p[1]] = {'tableName':p[1],'time':p[3],'unit':names['unit'],'currency':names['currency']
                           ,'company':names['company'],'table':'','tableBegin':True,'tableEnd':False}
            print(names[p[1]])

        def p_fetchtable_searchealy(p):
            '''fetchtable : TABLE optional UNIT optional '''
            print("search",p[1],p[3])
            names['unit'] = p[3].split(':')[-1].split('：')[-1]
            names[p[1]] = {'tableName':p[1],'unit':names['unit'],'currency':names['currency']
                           ,'table':'','tableBegin':True,'tableEnd':False}
            print(names[p[1]])

        def p_fetchtable_skiptime(p):
            '''fetchtable : TABLE optional TIME TIME '''
            #去掉主要会计数据的表头
            print(p[1])

        def p_fetchtable_skiptime(p):
            '''fetchtable : TABLE optional TIME TIME '''
            #去掉合并资产负债表项目
            print(p[1])

        def p_fetchtable_skipterm(p):
            '''fetchtable : TABLE term '''
            print(p[1])

        def p_fetchtable_skipterm(p):
            '''fetchtable : TABLE HEADER '''
            print(p[1])

        def p_skipword_group(p):
            '''skipword : '(' skipword ')'
                        | '（' skipword '）' '''
            p[0] = p[2]

        def p_skipword(p):
            '''skipword : useless skipword
                        | term skipword
                        | skipword DISCARD
                        | useless
                        | term '''
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
                names['currency'] = p[1].split(':')[-1].split('：')[-1]
            if p.slice[1].type == 'COMPANY':
                names['company'] = p[1]
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

    def doWork(self,docParser):
        for data in docParser:
            self._page = data
            self.parser.parse(docParser._get_text(data).replace(' ',''))

        '''
        #item = 83,6,120,111,71,149,110,4,38
        item = 108
        data = docParser._get_item(item)
        text = docParser._get_text(data)
        print(text)
        self.lexer.input(text)
        for token in self.lexer:
            print(token)
        self.parser.parse(text,lexer=self.lexer,debug=True)
        '''
        docParser._close()

def create_object(gConfig,docParser=None):
    interpreter=interpretAccounting(gConfig,docParser)
    interpreter.initialize()
    return interpreter

