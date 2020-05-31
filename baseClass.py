# -*- coding: utf-8 -*-
# @Time    : 9/25/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserBaseClass.py


#数据读写处理的基类
class baseClass():
    def __init__(self):
        self._data = list()
        self._index = 0
        self._length = len(self._data)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = self._data[self._index]
        except IndexError:
            raise StopIteration
        self._index += 1
        return data

    def __getitem__(self, item):
        return self._data[item]

    def _get_item(self,item):
        return self._data[item]

    def _load_data(self):
        pass

    def _get_text(self,page):
        pass

    def _get_tables(self,page):
        pass

    def _write_table(self,table):
        pass

    def _close(self):
        pass

