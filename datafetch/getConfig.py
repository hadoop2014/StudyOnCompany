#!/usr/bin/env Python
# coding   : utf-8
import configparser
import platform
import json
import os


def get_config(config_file='',config_file_base='config_directory/configbase.ini'):
    assert os.path.exists(config_file_base),"%s is not exists" % config_file_base
    assert config_file =='' or os.path.exists(config_file),"%s is not exists" % config_file

    parser=configparser.ConfigParser()
    if platform.system() == 'Windows':
        parser.read([config_file_base,config_file],encoding='utf-8-sig')
    else:
        parser.read([config_file_base,config_file],encoding='utf-8')
    # get the ints, floats and strings
    _conf_ints=_conf_floats=_conf_strings=_conf_bools=_conf_lists=_conf_sets=_conf_attrs=[]
    _conf_dicts = {}
    for section in parser.sections():
        if section == "int":
            _conf_ints = [(key, int(value)) for key, value in parser.items(section)]
        elif section == "floats":
            _conf_floats = [(key, float(value)) for key, value in parser.items(section)]
        elif section == "strings":
            _conf_strings = [(key, str(value)) for key, value in parser.items(section)]
        elif section == "bools":
            _conf_bools = [(key,parser.getboolean('bools',key)) for key, value in parser.items(section)]
        elif section == "lists":
            _conf_lists = [(key,eval(str(value))) for key, value in parser.items(section)]
        elif section == "sets":
            _conf_sets = [(key,list(map(str.strip,str(value).split(',')))) for key, value in parser.items(section)] #去掉空格
        elif section == "attrs":
            _conf_attrs = [(key, int(value)) for key, value in parser.items('attrs')]
        else:
            _conf_dicts = dict({section:dict(parser[section])},**_conf_dicts)

    return dict(_conf_ints + _conf_floats + _conf_strings + _conf_bools +
                _conf_lists + _conf_sets + _conf_attrs,**_conf_dicts)


def get_config_json_base(config_base_json = 'config_directory/interpreterBase.json'):
    assert os.path.exists(config_base_json), "%s is not exist,you must create first!" % config_base_json
    with open(config_base_json, encoding='utf-8') as json_file:
        config_json_base = json.load(json_file)
    return config_json_base


def get_config_json(config_file_json,config_base_json = 'config_directory/interpreterBase.json'):
    assert os.path.exists(config_file_json),"%s is not exist,you must create first!" % config_file_json
    assert os.path.exists(config_base_json),"%s is not exist,you must create first!" % config_base_json
    with open(config_file_json, encoding='utf-8') as json_file:
        config_json = json.load(json_file)
    with open(config_base_json, encoding='utf-8') as json_file:
        config_json_base = json.load(json_file)
    return config_json,config_json_base


def get_check_book(check_book_file = 'config_directory/checkbook.json'):
    if os.path.exists(check_book_file):
        with open(check_book_file, encoding='utf-8') as check_f:
            check_book = json.load(check_f)
    else:
        raise ValueError("%s is not exist,you must create first!" % check_book_file)
    assert check_book is not None,'check_book must not be None'
    return check_book