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

        #def t_PERCENTAGE(t):
        #    '\\d+[.\\d*]*%'
        #    t.value = int(t.value[:-1])*0.01
        #    print(t.value)
        #    return t

        #def t_NUMBER(t):
        #    r'\d+'
        #    t.value = int(t.value)
        #    return t


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
            ('right', 'UMINUS'),
        )

        # dictionary of names
        names = {}

        #def p_statement_assign(p):
        #    'statement : NAME "=" expression'
        #    names[p[1]] = p[3]


        def p_statement_expr(p):
            'statement : expression'
            print(p[1])

        def p_expression_search(p):
            '''expression : TABLE expression'''
            print("search",p[1],p[2])
            names[p[1]] = {'time':p[2],'unit':names['unit'],'currency':names['currency']}
            print(names[p[1]])

        #def p_expression_binop(p):
        #    '''expression : expression '+' expression
        #                  | expression '-' expression
        #                  | expression '*' expression
        #                  | expression '/' expression'''
        #    if p[2] == '+':
        #        p[0] = p[1] + p[3]
        #    elif p[2] == '-':
        #        p[0] = p[1] - p[3]
        #    elif p[2] == '*':
        #        p[0] = p[1] * p[3]
        #    elif p[2] == '/':
        #        p[0] = p[1] / p[3]
        #def p_expression_discardshiftl(p):
        #    'expression : DISCARD expression'
        #    p[0] = p[2]  #跳过DISCARD
        #    print(p[0])

        #def p_expression_discardshiftr(p):
        #    'expression : expression DISCARD'
        #    #p[0] = p[2]  #跳过DISCARD
        #    print(p[0])

        def p_expression_discard(p):
            'expression : DISCARD'
            #p[0] = p[2]  #跳过DISCARD
            print(p[0])

        def p_expression_unitshiftl(p):
            'expression : UNIT expression'
            #p[0] = p[2]  #跳过DISCARD
            names['unit'] = p[1]
            print(p[0])

        def p_expression_timeshiftl(p):
            'expression : TIME expression'
            p[0] = p[1]
            print(p[0])

        def p_expression_header(p):
            'expression : HEADER'
            p[0] = p[1]
            print(p[0])

        def p_expression_percentage(p):
            'expression : PERCENTAGE'
            p[0] = p[1]
            print(p[0])

        def p_expression_value(p):
            'expression : VALUE'
            p[0] = float(p[1].replace(',',''))
            print(p[0])

        def p_expression_uminus(p):
            "expression : '-' expression %prec UMINUS"
            p[0] = -int(p[2])


        def p_expression_group(p):
            "expression : '(' expression ')'"
            p[0] = p[2]


        def p_expression_number(p):
            "expression : NUMBER"
            p[0] = p[1]
            print(p[0])



        #def p_expression_name(p):
        #    "expression : NAME"
        #    try:
        #        p[0] = names[p[1]]
        #    except LookupError:
        #        print("Undefined name '%s'" % p[1])
        #        p[0] = 0

        #def p_expression_unit(p):
        #    'expression : UNIT expression'
        #    names['unit'] = p[1]
        #    p[0] = p[1]

        def p_expression_currency(p):
            'expression : CURRENCY'
            names['currency'] = p[1]
            p[0] = p[1]

        def p_error(p):
            if p:
                print("Syntax error at '%s:%s'" % (p.value,p.type))
            else:
                print("Syntax error at EOF")


        # Build the docparser
        self.parser = yacc.yacc(outputdir=self.working_directory)

    def doWork(self,docParser):
        #for data in docParser:
        #    self.parser.parse(docParser._get_text(data).replace(' ',''))

        item = 76
        data = docParser._get_item(item)
        text = docParser._get_text(data)
        print(text)
        self.parser.parse(text)
        self.lexer.input(text)
        for token in self.lexer:
            print(token)

        docParser._close()

def create_object(gConfig):
    interpreter=interpretAccounting(gConfig)
    interpreter.initialize()
    return interpreter

