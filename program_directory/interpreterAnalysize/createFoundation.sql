drop table if exists 财务分析基础表;
create table if not exists 财务分析基础表 (
    --ID INTEGER PRIMARY KEY AUTOINCREMENT,
    报告时间 DATE NOT NULL,
    公司代码 INTEGER NOT NULL,
    报告类型 CHAR(20) NOT NULL,
    公司简称 CHAR(10),
    公司名称 CHAR(50),
    办公地址 CHAR(10),
    行业分类 CHAR(10),
    在职员工的数量合计 NUMERIC,
    应付职工薪酬 NUMERIC,
    营业收入 NUMERIC,
    归属于上市公司股东的净利润 NUMERIC,
    归属于上市公司股东的扣除非经常性损益的净利润 NUMERIC,
    经营活动产生的现金流量净额 NUMERIC,
    --归属于上市公司股东的扣除非经常性损益的净利润 NUMERIC,
    总资产 NUMERIC,
    现金分红金额（含税） NUMERIC,
    现金分红总额（含其他方式）占合并报表中归属于上市公司普通股股东的净利润的比率 NUMERIC,
    营业成本 NUMERIC,
    投资收益 NUMERIC,
    三、营业利润 NUMERIC,
    四、利润总额 NUMERIC,
    所得税费用 NUMERIC,
    五、净利润 NUMERIC,
    销售费用 NUMERIC,
    管理费用 NUMERIC,
    财务费用 NUMERIC,
    研发费用 NUMERIC,
    --"无形资产-内部研发","资产减值准备","折旧和摊销","所得税税率",
    期末总股本 NUMERIC,
    固定资产 NUMERIC,
    在建工程 NUMERIC,
    --c.无形资产-土地使用权",
    投资性房地产 NUMERIC,
    商誉 NUMERIC,
    预收款项 NUMERIC,
    应付票据及应付账款 NUMERIC,
    应付账款 NUMERIC,
    预付款项 NUMERIC,
    应收账款 NUMERIC,
    应收票据 NMERIC,--"应收票据（上期）",
    --流动资产合计 NUMERIC,
    流动负债合计 NUMERIC,
    存货 NUMERIC,
    货币资金 NUMERIC,
    短期借款 NUMERIC,
    一年内到期的非流动负债 NUMERIC,--"有息债券(短期借款＋一年内到期的非流动负)","应收票据增加额（本期应收票据余额-上期应收票据余额）",
    长期借款 NUMERIC,
    应付债券 NUMERIC,--"流动比率","速动比率","有息负债"
    流动资产合计 NUMERIC,--"负债合计",
    销售商品、提供劳务收到的现金 NUMERIC,
    [六、期末现金及现金等价物余额] NUMERIC,
    五、现金及现金等价物净增加额 NUMERIC--f.现金收入,
    --b.归属于上市公司股东的净利润,replace(b.营业收入,',',''),
    --round(replace(b.营业收入,',','')/replace(a.在职员工的数量合计,',',''),0) as 人均营业收入
);

insert into 财务分析基础表
select
       a.报告时间,
       a.公司代码,
       a.报告类型,
       a.公司简称,
       a.公司名称,
       a.办公地址,
       a.行业分类,
       a.在职员工的数量合计,
       c.应付职工薪酬,
       b.营业收入,
       b.归属于上市公司股东的净利润,
       b.归属于上市公司股东的扣除非经常性损益的净利润,
       b.经营活动产生的现金流量净额,
       --b.归属于上市公司股东的扣除非经常性损益的净利润,
       b.总资产,
       d.现金分红金额（含税）,
       d.现金分红总额（含其他方式）占合并报表中归属于上市公司普通股股东的净利润的比率,
       e.营业成本,
       e.投资收益,
       e.三、营业利润,
       e.四、利润总额,
       e.所得税费用,
       e.五、净利润,
       e.销售费用,
       e.管理费用,
       e.财务费用,
       e.研发费用,
       --"无形资产-内部研发","资产减值准备","折旧和摊销","所得税税率",
       b.期末总股本,
       c.固定资产,
       c.在建工程,
       --c.无形资产-土地使用权",
       c.投资性房地产,
       c.商誉,
       c.预收款项,
       c.应付票据及应付账款,
       c.应付账款,
       c.预付款项,
       c.应收账款,
       c.应收票据,
       --"应收票据（上期）",
       --c.流动资产合计,
       c.流动负债合计,
       c.存货,
       c.货币资金,
       c.短期借款,
       c.一年内到期的非流动负债,
       --"有息债券(短期借款＋一年内到期的非流动负)","应收票据增加额（本期应收票据余额-上期应收票据余额）",
       c.长期借款,
       c.应付债券,
       --"流动比率","速动比率","有息负债"
       c.流动资产合计,--"负债合计",
       f.销售商品、提供劳务收到的现金,
       f.[六、期末现金及现金等价物余额],
       f.五、现金及现金等价物净增加额
       --f.现金收入,
       --b.归属于上市公司股东的净利润,replace(b.营业收入,',',''),
       --round(replace(b.营业收入,',','')/replace(a.在职员工的数量合计,',',''),0) as 人均营业收入
from 关键数据表 a
  left join 主要会计数据 b
  left join 合并资产负债表 c
  left join 普通股现金分红情况表 d
  left join 合并利润表 e
  left join 合并现金流量表 f
where (a.报告时间 = b.报告时间 and a.公司代码 = b.公司代码 and a.报告类型 = b.报告类型)
  and (a.报告时间 = c.报告时间 and a.公司代码 = c.公司代码 and a.报告类型 = c.报告类型)
  and (a.报告时间 = d.报告时间 and a.公司代码 = d.公司代码 and a.报告类型 = d.报告类型)
  and (a.报告时间 = e.报告时间 and a.公司代码 = e.公司代码 and a.报告类型 = e.报告类型)
  and (a.报告时间 = f.报告时间 and a.公司代码 = f.公司代码 and a.报告类型 = f.报告类型)
order by a.报告时间,a.公司代码,a.报告类型;

CREATE INDEX IF NOT EXISTS [财务分析基础表索引] on [财务分析基础表] (
    报告时间,
    公司代码,
    报告类型
);