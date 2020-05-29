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
        #tokens = (
        #    'NAME',
        #    'NUMBER',
        #)
        tokens = self.tokens

        #literals = ['=', '+', '-', '*', '/', '(', ')']
        literals = self.literals

        # Tokens

        #"TABLE": "主要会计数据|合并资产负债表|合并利润表|合并现金流量表",
        #"NUMBER": "d+",
        #"UNIT": "元|千元|万元|百万元|千万元|亿",
        #"CURRENCY": "人民币|美元|欧元|[\\u4E00-\\u9FA5]元",
        #"COMPANY": ".*?[\\u4E00-\\u9FA5]+公司",
        #"TIME": "",
        #"HEADER": "科目|项目|分红年度",
        #"PERCENTAGE": "",
        #"NAME": "r'[a-zA-Z_][a-zA-Z0-9_]*'",
        t_TABLE = self.dictTokens['TABLE']
        #t_NUMBER = self.dictTokens['NUMBER']
        t_UNIT = self.dictTokens['UNIT']
        t_CURRENCY = self.dictTokens['CURRENCY']
        t_COMPANY = self.dictTokens['COMPANY']
        t_TIME = self.dictTokens['TIME']
        t_HEADER = self.dictTokens['HEADER']
        t_PERCENTAGE = self.dictTokens['PERCENTAGE']
        #t_NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'
        t_NAME = self.dictTokens['NAME']

        def t_NUMBER(t):
            r'\d+'
            t.value = int(t.value)
            return t


        t_ignore = " \t\n"

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
            ('left', '+', '-'),
            ('left', '*', '/'),
            ('right', 'UMINUS'),
        )

        # dictionary of names
        names = {}

        def p_statement_search(p):
            'statement : TABLE'
            print(p[1])

        def p_statement_assign(p):
            'statement : NAME "=" expression'
            names[p[1]] = p[3]


        def p_statement_expr(p):
            'statement : expression'
            print(p[1])


        def p_expression_binop(p):
            '''expression : expression '+' expression
                          | expression '-' expression
                          | expression '*' expression
                          | expression '/' expression'''
            if p[2] == '+':
                p[0] = p[1] + p[3]
            elif p[2] == '-':
                p[0] = p[1] - p[3]
            elif p[2] == '*':
                p[0] = p[1] * p[3]
            elif p[2] == '/':
                p[0] = p[1] / p[3]


        def p_expression_uminus(p):
            "expression : '-' expression %prec UMINUS"
            p[0] = -p[2]


        def p_expression_group(p):
            "expression : '(' expression ')'"
            p[0] = p[2]


        def p_expression_number(p):
            "expression : NUMBER"
            p[0] = p[1]


        def p_expression_name(p):
            "expression : NAME"
            try:
                p[0] = names[p[1]]
            except LookupError:
                print("Undefined name '%s'" % p[1])
                p[0] = 0


        def p_error(p):
            if p:
                print("Syntax error at '%s'" % p.value)
            else:
                print("Syntax error at EOF")

        # Build the docparser
        self.parser = yacc.yacc(outputdir=self.working_directory)

    def doWork(self,docParser):
        for data in docParser:
            self.parser.parse(docParser._get_text(data))
        docParser._close()

def create_object(gConfig):
    interpreter=interpretAccounting(gConfig)
    interpreter.initialize()
    return interpreter

