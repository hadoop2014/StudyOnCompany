# -----------------------------------------------------------------------------
# interpretCalc.py
#
# A simple calculator with variables.   This is from O'Reilly's
# "Lex and Yacc", p. 63.
# -----------------------------------------------------------------------------
from interpreter.interpretBaseClass import *


class interpretAccounting(interpretBase):
    def __init__(self,gConfig):
        super(interpretAccounting,self).__init__(gConfig)
        #self.docParser = docParser
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
        #    #('right', 'PERCENTAGE','NUMBER'),
        #    ('left', '+', '-'),
        #    ('left', '*', '/'),
        #    ('left', 'NOTHING','(',')'),
        #    ('left', 'GROUP'),
            ('right', 'UMINUS'),
        )

        # dictionary of names
        names = {}

        def p_statement_statement(p):
            '''statement : statement expression
                         | expression '''
            p[0] = p[1]

        def p_statement_grouphalf(p):
            '''statement : statement ')'
                         | statement '）' '''
            p[0] = p[1]

        def p_statement_search(p):
            '''statement : TABLE TIME UNIT CURRENCY expression'''
            print("search",p[1],p[2])
            names['unit'] = p[3].split(':')[-1].split('：')[-1]
            names['currency'] = p[4].split(':')[-1].split('：')[-1]
            names[p[1]] = {'table':p[1],'time':p[2],'unit':names['unit'],'currency':names['currency']}
            print(names[p[1]])

        def p_statement_searchealy(p):
            '''statement : TABLE UNIT CURRENCY expression'''
            print("search",p[1],p[2])
            names['unit'] = p[2].split(':')[-1].split('：')[-1]
            names['currency'] = p[3].split(':')[-1].split('：')[-1]
            names[p[1]] = {'table':p[1],'unit':names['unit'],'currency':names['currency']}
            print(names[p[1]])

        #def p_statemnt_tableshiftr(p):
        #    '''statement : TABLE expression'''
        #    p[0] = p[1]

        def p_expression_reduce(p):
            '''expression : expression expression
                          | term
                          | nothing '''
            p[0] = p[1]

        def p_expression_group(p):
            '''expression : '(' expression ')'
                          | '（' expression '）' '''
            p[0] = p[2]

        def p_nothing_reduce(p):
            '''nothing : '-' nothing '''
            p[0] = p[1]

        def p_nothing(p):
            '''nothing :  PUNCTUATION
                       | DISCARD
                       | WEBSITE
                       | EMAIL
                       | COMPANY
                       | NAME
                       | TIME
                       | HEADER
                       | CURRENCY
                       | UNIT
                       | TABLE
                       | '-'
                       | '%' '''
            print('nothing ',p[1])

        def p_term_percentage(p):
            '''term : NUMERIC '%' '''
            p[0] = round(float(p[1]) * 0.01,4)
            print('percentage',p[0])

        def p_term_numeric(p):
            '''term : NUMERIC'''
            if p[1].find('.') < 0 :
                p[0] = int(p[1].replace(',',''))
            else:
                p[0] = float(p[1].replace(',',''))
            print('value',p[0],p[1])

        def p_term_uminus(p):
            '''term : '-' term %prec UMINUS'''
            p[0] = -p[2]

        def p_term_group(p):
            '''term : '(' term ')' '''
            p[0] = -p[2]  #财务报表中()表示负值

        def p_error(p):
            if p:
                print("Syntax error at '%s:%s'" % (p.value,p.type))
            else:
                print("Syntax error at EOF")


        # Build the docparser
        self.parser = yacc.yacc(outputdir=self.working_directory)

    def doWork(self,docParser):
        for data in docParser:
            self.parser.parse(docParser._get_text(data).replace(' ',''))

        '''
        #item = 83,6,120,111
        item = 123
        data = docParser._get_item(item)
        text = docParser._get_text(data)
        print(text)
        self.lexer.input(text)
        for token in self.lexer:
            print(token)
        self.parser.parse(text,lexer=self.lexer,debug=True)
        '''
        docParser._close()

def create_object(gConfig):
    interpreter=interpretAccounting(gConfig)
    interpreter.initialize()
    return interpreter

