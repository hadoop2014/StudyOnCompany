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
            ('left', 'NOTHING','(',')'),
            ('left', 'GROUP'),
            ('right', 'UMINUS'),
        )

        # dictionary of names
        names = {}

        def p_statement_statement(p):
            '''statement : statement statement'''
            #print(p[1],p[2])
            p[0] = p[1]
            #print(p[0])

        def p_statement_group(p):
            '''statement : '(' statement ')'
                        | '（' statement '）' '''
            #print(p[1],p[2])
            p[0] = p[2]


        def p_statement_expr(p):
            '''statement : expression'''
            p[0] = p[1]

        def p_expression_search(p):
            '''expression : TABLE TIME UNIT expression'''
            print("search",p[1],p[2])
            names['unit'] = p[3].split(':')[-1].split('：')[-1]
            names[p[1]] = {'table':p[1],'time':p[2],'unit':names['unit'],'currency':names['currency']}
            print(names[p[1]])

        def p_expression_searchealy(p):
            '''expression : TABLE UNIT expression'''
            print("search",p[1],p[2])
            names['unit'] = p[2].split(':')[-1].split('：')[-1]
            names[p[1]] = {'table':p[1],'unit':names['unit'],'currency':names['currency']}
            print(names[p[1]])

        def p_expression_tableshiftr(p):
            '''expression : TABLE expression'''
            p[0] = p[1]

        def p_expression_group(p):
            '''expression : '(' expression ')'
                          | '（' expression '）'
                          | '(' '%' ')'
                          | '（' '%' '）' '''
            p[0] = p[2]

        def p_expression_currency(p):
            '''expression : CURRENCY'''
            names['currency'] = p[1].split(':')[-1].split('：')[-1]
            p[0] = p[1]
            print(p[0])

        def p_expression_unit(p):
            '''expression : UNIT'''
            names['unit'] = p[1].split(':')[-1].split('：')[-1]
            p[0] = p[1]
            print(p[0])

        def p_expression_discardshiftr(p):
            '''expression : expression DISCARD'''
            p[0] = p[1]

        def p_expression_timeshiftr(p):
            '''expression : expression TIME'''
            p[0] = p[1]

        def p_expression_nothing(p):
            '''expression : nothing'''
            p[0] = p[1]

        def p_nothing(p):
            '''nothing : PUNCTUATION
                       | DISCARD
                       | WEBSITE
                       | EMAIL
                       | COMPANY
                       | NAME
                       | TIME
                       | HEADER
                       | UNIT
                       | '-'
                       | NUMERIC ')' %prec NOTHING '''

            print('nothing ',p[1])

        def p_expression_term(p):
            '''expression : term %prec GROUP'''
            p[0] = p[1]

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
            '''term : '(' term ')' %prec GROUP '''
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
        item = 111
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

