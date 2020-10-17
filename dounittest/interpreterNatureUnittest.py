import unittest
from interpreterAssemble import InterpreterAssemble

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.interpreter = InterpreterAssemble().get_interpreter_nature(unittestIsOn=True)
        self.run_interpreter_yacc()

    def run_interpreter_yacc(self):
        input = ''
        input = input + ' 批量 运行 财务报表解析'
        input = input + ' 单次 运行 财务报表解析 '
        input = input + ' 参数配置{公司简称: 华侨城A,华侨城Ａ}'
        self.interpreter.lexer.input(input)
        for tok in self.interpreter.lexer:
            print(tok)
        #self.interpreter.initialize()
        self.interpreter.parser.parse(input,lexer=self.interpreter.lexer,debug=True,tracking=True)
if __name__ == '__main__':
    unittest.main()
