drop table if exists 财务分析基础表;
create table if not exists 财务分析基础表 (
    --ID INTEGER PRIMARY KEY AUTOINCREMENT,
    报告时间 DATE NOT NULL,
    公司代码 INTEGER NOT NULL,
    报告类型 CHAR(20) NOT NULL,
    公司简称 CHAR(10),
    公司名称 CHAR(50),
    公司地址 CHAR(10),
    行业分类 CHAR(10),
    在职员工的数量合计 NUMERIC,
    支付给职工及为职工支付的现金 NUMERIC,
    应付职工薪酬（期末余额） NUMERIC,
    应付职工薪酬（期初余额） NUMERIC,
    营业收入 NUMERIC,
    归属于上市公司股东的净利润 NUMERIC,
    归属于上市公司股东的扣除非经常性损益的净利润 NUMERIC,
    经营活动产生的现金流量净额 NUMERIC,
    归属于上市公司股东的净资产 NUMERIC,
    总资产 NUMERIC,
    现金分红金额（含税） NUMERIC,
    现金分红金额占合并报表中归属于上市公司普通股股东的净利润的比率 NUMERIC,
    营业收入（上期） NUMERIC,
    归属于上市公司股东的净利润（上期） NUMERIC,
    归属于上市公司股东的扣除非经常性损益的净利润（上期） NUMERIC,
    经营活动产生的现金流量净额（上期） NUMERIC,
    归属于上市公司股东的净资产（上期） NUMERIC,
    总资产（上期） NUMERIC,
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
    资本化研发投入 NUMERIC,
    资产减值准备 NUMERIC,
    固定资产折旧、油气资产折耗、生产性生物资产折旧 NUMERIC,
    无形资产摊销 NUMERIC,
    长期待摊费用摊销 NUMERIC,
    --"无形资产-内部研发","所得税税率",
    期末总股本 NUMERIC,
    固定资产 NUMERIC,
    在建工程 NUMERIC,
    土地使用权 NUMERIC,
    投资性房地产 NUMERIC,
    商誉 NUMERIC,
    预收款项 NUMERIC,
    应付票据及应付账款 NUMERIC,
    应付账款 NUMERIC,
    预付款项 NUMERIC,
    应收账款 NUMERIC,
    应收票据 NMERIC,
    流动资产合计 NUMERIC,
    负债合计 NUMERIC,
    流动负债合计 NUMERIC,
    存货 NUMERIC,
    货币资金 NUMERIC,
    短期借款 NUMERIC,
    一年内到期的非流动负债 NUMERIC,
    长期借款 NUMERIC,
    应付债券 NUMERIC,
    销售商品、提供劳务收到的现金 NUMERIC,
    [六、期末现金及现金等价物余额] NUMERIC,
    五、现金及现金等价物净增加额 NUMERIC
);

insert into 财务分析基础表
select
    a.报告时间,
    a.公司代码,
    a.报告类型,
    a.公司简称,
    a.公司名称,
    a.公司地址,
    a.行业分类,
    a.在职员工的数量合计,
    f.支付给职工及为职工支付的现金,
    c.应付职工薪酬 as 应付职工薪酬（期末余额）,
    c.应付职工薪酬（期初余额）,
    b.营业收入,
    b.归属于上市公司股东的净利润,
    b.归属于上市公司股东的扣除非经常性损益的净利润,
    b.经营活动产生的现金流量净额,
    b.归属于上市公司股东的净资产,
    b.总资产,
    d.现金分红金额（含税）,
    d.现金分红金额占合并报表中归属于上市公司普通股股东的净利润的比率,
    b.营业收入（上期）,
    b.归属于上市公司股东的净利润（上期）,
    b.归属于上市公司股东的扣除非经常性损益的净利润（上期）,
    b.经营活动产生的现金流量净额（上期）,
    b.归属于上市公司股东的净资产（上期）,
    b.总资产（上期）,
    e.营业成本,
    e.投资收益,
    e.三、营业利润,
    e.四、利润总额,
    e.所得税费用,
    e.五、净利润,
    e.销售费用,
    e.管理费用,
    e.财务费用,
    case when a.本期费用化研发投入修正 != '' then a.本期费用化研发投入修正 else e.研发费用 end as 研发费用,
    --case when a.本期资本化研发投入 is NULL  or a.本期资本化研发投入 = '' then 0 else a.本期资本化研发投入 end as 资本化研发投入,
    a.本期资本化研发投入修正 as 资本化研发投入,
    g.资产减值准备 NUMERIC,
    g.固定资产折旧、油气资产折耗、生产性生物资产折旧 NUMERIC,
    g.无形资产摊销 NUMERIC,
    g.长期待摊费用摊销 NUMERIC,
    --"无形资产-内部研发","所得税税率",
    b.期末总股本,
    c.固定资产,
    c.在建工程,
    h.期末账面价值 as 土地使用权,
    c.投资性房地产,
    c.商誉,
    c.预收款项,
    c.应付票据及应付账款,
    c.应付账款,
    c.预付款项,
    c.应收账款,
    c.应收票据,
    c.流动资产合计,
    c.负债合计,
    c.流动负债合计,
    c.存货,
    c.货币资金,
    c.短期借款,
    c.一年内到期的非流动负债,
    c.长期借款,
    c.应付债券,
    f.销售商品、提供劳务收到的现金,
    f.[六、期末现金及现金等价物余额],
    f.五、现金及现金等价物净增加额
from
(
    select x.*,
        case when 本期费用化研发投入 = '' and 研发投入金额 != '' and 本期资本化研发投入 != ''
            then round(replace(研发投入金额,',','') - replace(本期资本化研发投入,',',''),2) else 本期费用化研发投入 end as 本期费用化研发投入修正,
        case when 本期资本化研发投入 = '' and 研发投入金额 != '' and 本期费用化研发投入 != ''
            then round(replace(研发投入金额,',','') - replace(本期费用化研发投入,',',''),2) else 本期资本化研发投入 end as 本期资本化研发投入修正
    from 关键数据表 x
    where x.在职员工的数量合计 != '' and x.报告类型 = '年度报告'
)a
left join
(
    select x.报告时间,
        x.公司代码,
        x.公司简称,
        x.报告类型,
        x.营业收入,
        x.归属于上市公司股东的净利润,
        x.归属于上市公司股东的扣除非经常性损益的净利润,
        x.经营活动产生的现金流量净额,
        case when x.归属于上市公司股东的净资产 is not NULL
            then x.归属于上市公司股东的净资产 else z.所有者权益（或股东权益）合计 end as 归属于上市公司股东的净资产,
        case when x.总资产 is not NULL then x.总资产 else z.负债和所有者权益（或股东权益）总计 end as 总资产,
        case when x.期末总股本 is not NULL then x.期末总股本 else z.实收资本（或股本） end as 期末总股本,
        y.营业收入 as 营业收入（上期）,
        y.归属于上市公司股东的净利润 as 归属于上市公司股东的净利润（上期）,
        y.归属于上市公司股东的扣除非经常性损益的净利润 as 归属于上市公司股东的扣除非经常性损益的净利润（上期）,
        y.经营活动产生的现金流量净额 as 经营活动产生的现金流量净额（上期）,
        y.归属于上市公司股东的净资产 as 归属于上市公司股东的净资产（上期）,
        y.总资产 as 总资产（上期）
    from 主要会计数据 x
    left join 主要会计数据 y
    left join 合并资产负债表 z
    where (x.报告时间 - y.报告时间 = 1 and x.报告类型 = y.报告类型 and x.公司代码 = y.公司代码)
        and (x.报告时间 = z.报告时间 and x.公司代码 = z.公司代码 and x.报告类型 = z.报告类型)
)b
left join
(
    select x.*,
        y.应付职工薪酬 as 应付职工薪酬（期初余额）
    from 合并资产负债表 x
    left join 合并资产负债表 y
    where x.报告时间 - y.报告时间 = 1
        and x.报告类型 = y.报告类型
        and x.公司代码 = y.公司代码
)c
left join 普通股现金分红情况表 d
left join 合并利润表 e
left join 合并现金流量表 f
left join 现金流量表补充资料 g
left join 无形资产情况 h
where (a.报告时间 = b.报告时间 and a.公司代码 = b.公司代码 and a.报告类型 = b.报告类型)
    and (a.报告时间 = c.报告时间 and a.公司代码 = c.公司代码 and a.报告类型 = c.报告类型)
    and (a.报告时间 = d.报告时间 and a.公司代码 = d.公司代码 and a.报告类型 = d.报告类型)
    and (a.报告时间 = e.报告时间 and a.公司代码 = e.公司代码 and a.报告类型 = e.报告类型)
    and (a.报告时间 = f.报告时间 and a.公司代码 = f.公司代码 and a.报告类型 = f.报告类型)
    and (a.报告时间 = g.报告时间 and a.公司代码 = g.公司代码 and a.报告类型 = g.报告类型)
    and (a.报告时间 = h.报告时间 and a.公司代码 = h.公司代码 and a.报告类型 = h.报告类型 and h.项目 = '土地使用权')
order by a.报告时间,a.公司代码,a.报告类型;

CREATE INDEX IF NOT EXISTS [财务分析基础表索引] on [财务分析基础表] (
    报告时间,
    公司代码,
    报告类型
);

select *
from 财务分析基础表 a
order by a.公司代码, a.报告时间 desc,a.报告类型;

