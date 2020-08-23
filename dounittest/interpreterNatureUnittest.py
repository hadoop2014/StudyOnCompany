import unittest
from interpreterAssemble import InterpreterAssemble

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.interpreter = InterpreterAssemble().get_interpreter_nature(unittestIsOn=True)
        #self.run_interpreter_lexer()
        self.run_interpreter_yacc()

    def run_interpreter_lexer(self):
        input = ''
        input = input + ' 批量运行 财务报表解析'
        input = input + ' ##批量运行 财务报表解析'
        input = input + ' 单次运行 财务报表解析 '
        self.interpreter.lexer.input(input)
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(-,'-',1,1)")

    def run_interpreter_yacc(self):
        input = ''
        input = input + ' ##批量运行 财务报表解析\n'
        input = input + ' 批量运行 财务报表解析'
        input = input + ' 单次运行 财务报表解析 '
        self.interpreter.lexer.input(input)
        for tok in self.interpreter.lexer:
            print(tok)
        #self.interpreter.initialize()
        self.interpreter.parser.parse(input,lexer=self.interpreter.lexer,debug=True,tracking=True)
if __name__ == '__main__':
    unittest.main()
