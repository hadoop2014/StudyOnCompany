import configparser
import platform

def get_config(config_file='',config_file_base='config_directory/configbase.txt'):
    parser=configparser.ConfigParser()
    if platform.system() == 'Windows':
        parser.read([config_file_base,config_file],encoding='utf-8-sig')
    else:
        parser.read([config_file_base,config_file],encoding='utf-8')
    # get the ints, floats and strings
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    _conf_bools = [(key,parser.getboolean('bools',key)) for key, value in parser.items('bools')]
    _conf_lists = [(key,eval(str(value))) for key, value in parser.items('lists')]
    _conf_sets = [(key,list(map(str.strip,str(value).split(',')))) for key, value in parser.items('sets')] #去掉空格
    _conf_attrs = [(key, int(value)) for key, value in parser.items('attrs')]
    return dict(_conf_ints + _conf_floats + _conf_strings + _conf_bools +
                _conf_lists + _conf_sets + _conf_attrs)