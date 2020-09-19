#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/20/2020 5:03 PM
# @Author  : wu.hao
# @File    : crawlFinance.py
# @Note    : 用于从互联网上爬取财务报表


from interpreterCrawl.webcrawl.crawlBaseClass import *


class CrawlFinance(CrawlBase):
    def __init__(self,gConfig):
        super(CrawlFinance, self).__init__(gConfig)


    def initialize(self,gConfig = None):
        if gConfig is not None:
            self.gConfig = gConfig
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self.clear_logging_directory(self.logging_directory)


def create_object(gConfig):
    parser=CrawlFinance(gConfig)
    parser.initialize()
    return parser
