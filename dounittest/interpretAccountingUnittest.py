import unittest
from datafetch import getConfig
from interpreter import interpretAccounting


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)
        self.run_interpreter()

    def run_interpreter(self):
        gConfig = getConfig.get_config('config_directory/configbase.txt')
        gConfigJson = getConfig.get_config_json('config_directory/interpretAccounting.json')
        gConfig.update({"gConfigJson": gConfigJson})
        interpreter = interpretAccounting.create_object(gConfig=gConfig)
        #test lexer
        input = ''
        #input = input + '1,370,249,543.17'
        #input = input + ' 45.01%'
        #input = input + '2008 年 12月5  日'
        input = input + 'www.1_0see.com'
        interpreter.lexer.input(input)
        for tok in interpreter.lexer:
            print(tok)
        #test yaac
        interpreter.parser.parse(input)

if __name__ == '__main__':
    unittest.main()
