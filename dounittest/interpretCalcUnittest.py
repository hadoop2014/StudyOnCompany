import unittest
from interpreter import interpretCalc
from datafetch import getConfig

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)
        self.run_interpreter()

    def run_interpreter(self):
        gConfig = getConfig.get_config('config_directory/configbase.txt')
        gConfigJson = getConfig.get_config_json('config_directory/interpretAccounting.json')
        gConfig.update({"gConfigJson": gConfigJson})
        interpreter = interpretCalc.create_object(gConfig=gConfig)
        while True:
            try:
                s = input('calc > ')
            except EOFError:
                break
            interpreter.lexer.input(s)
            while True:
                tok = interpreter.lexer.token()
                if not tok: break  # No more input
                print(tok)
            if not s: continue
            interpreter.parser.parse(s,debug=True)

if __name__ == '__main__':
    unittest.main()
