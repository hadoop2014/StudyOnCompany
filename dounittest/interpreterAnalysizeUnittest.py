import unittest
from interpreterAssemble import InterpreterAssemble


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.interpreter = InterpreterAssemble().interpreter_assemble('analysize',unittestIsOn=True)
        self.run_interpreter_yacc()

    def run_interpreter_yacc(self):
        input = ''
        input = input + "  #创建 财务分析综合表\n"
        input = input + "  创建 股票价格分析表"
        self.interpreter.lexer.input(input)
        for tok in self.interpreter.lexer:
            print(tok)
        self.interpreter.parser.parse(input, lexer=self.interpreter.lexer, debug=True, tracking=True)


if __name__ == '__main__':
    unittest.main()
