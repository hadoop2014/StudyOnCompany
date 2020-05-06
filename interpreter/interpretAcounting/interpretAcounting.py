
tokens = (
            'Specifications',#技术规格
            'Taxonomy',#分类标准
            'InstanceDocuments',#实例文档
            'StyleSheets',#样式单
)

#分类标准
taxonomys = (
                 'DocumentInformation',#文档信息:描述此次文档的制定日期、制定人、语言版本等相关的诸多信息；
                 'EntityInformation',# 实体信息:描述企业的名称、地址、所处行业等相关的信息；
                 'AccountantsReport',# 查核报告书:显示会计师出具报告的类型及内容，如会计师的姓名；
                 'BalanceSheet',#资产负债表:与通用的会计资产负债表一样，这里遵循的是US GAAP C&I，如“Cash”和“Long Term Debt”等；
                 'IncomeStatement',#损益表:包含与收入有关的信息，如 “Sales Revenues, Net”，“Income（Loss） from Continuing Operations”；
                 'StatementOfComprehensiveIncome',#综合损益表:包含与综合收益相关的陈述，如“Other Comprehensive Income”；'
                 'StatementOfStockholdersEquity',#股东权益变动表:包含与股东资产净值有关的陈述，如“Sale of Common Stock”；
                 'cashFlows',#现金流量表:描述和现金流量相关的信息；
                 'NotesToFinancialStatements',#财务报表附注:与传统方式大致相同，但是以XBRL标签来表达的，如“Significant Accounting Policies”；
                 'Supplemental Financial Information',#补充会计信息项目:显示由实体发布的补充会计信息；
                 'FinancialHighlights',#企业想强调的财务信息:显示由实体发布的重点会计信息；
                 'Management Discussion And Analysis',#企业运营讨论及分析:显示未来的运营策略信息。
)
#每个分类标准都包含一个模式文件和五个xml文件，其中模式文件是分类标准的核心，其中：
#　　taxonomy.xsd文件定义分类标准包含的项目及其类型信息，项目的其他信息在其他五个xml文件中定义；
#　　definition.xml定义从概念角度理解项目与项目之间的关系；
#　　calculation.xml定义从数据计算角度理解项目与项目之间的关系；
#　　1abel.xml文件定义项目的标签，该文件的信息确定了项目在财务报告中实际显示的名称；
#　　presentation.xml文件定义在财务报告中，统一父项目下所有子项目的显示顺序；
#　　reference.xml文件定义项目的参考信息，通过该文件，结合definition.xml文件的信息，我们准确理解项目的实际含义。

#实例文档
instanceDocument='''
<?xml version="1.0" encoding="UTF-8"?>
　　 <xbrli:xbrlxmlns:xbrli=http://www.xbrl.org/2003/instancexmlns:l
　　ink="http://www.xbrl.org/2003/linkbase" :xlink="http://www.w3.org/1
　　999/xlink":p0="http://www.fujitsu.com/xbrl/taxeditor/sail"xmlns:iso42
　　17="http://www.xbrl.org/2003/iso4217">
　　 <link:schemaRef xlink:type="simple" xlink:href="sail.xsd"/>
　　 <xbrli:context id="c1">
　　 <xbrli:entity>
　　 <xbrli:identifier scheme="http://www.fujitsu.com/xbrl/
　　taxeditor/sail">风帆公司资产负债表</xbrli:identifier>
　　 </xbrli:entity>
　　 <xbrli:period>
　　 <xbrli:startDate>2005-01-01</xbrli:startDate> <xbrli:endDate>
　　2005-13-31</xbrli:endDate>
　　 </xbrli:period>
　　 </xbrli:context>
　　 <xbrli:unit id="u1">
　　 <xbrli:measure>人民币元</xbrli:measure>
　　 </xbrli:unit>
　　 <p0:资产负债表>
　　 <p0: 流动资产 decimals="0" contextRef="c1" unitRef=
　　"u1">6 200 000</p0:流动资产>
　　 <p0: 货币资金 decimals="0" contextRef="c1"
　　unitRef="u1">3920000</p0:货币资金>
　　 <p0: 应收账款净值 decimals="0" contextRef="c1"
　　unitRef="u1">2 280 000</p0:应收账款净值>
　　 <p0: 存货 decimals="0" contextRef="c1" unitRef=
　　"u1">0</p0:存货>
　　 </p0:资产负债表>
　　 </xbrli:xbrl>
'''