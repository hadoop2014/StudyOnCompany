--参数{0}会被替换成报告类型,如: 年报,半年报,季报
drop table if exists {0}财务分析综合表;
create table if not exists {0}财务分析综合表 (
    报告时间 DATE NOT NULL,
    公司代码 INTEGER NOT NULL,
    报告类型 CHAR(20) NOT NULL,
    公司简称 CHAR(10),
    公司名称 CHAR(50),
    公司地址 CHAR(10),
    行业分类 CHAR(10),
    发布时间 DATE,
    公司投资等级 REAL,
    预测市值增长比值 REAL,
    在职员工的数量合计 REAL,
    支付给职工及为职工支付的现金 REAL,
    [应付职工薪酬（期末余额）] REAL,
    [应付职工薪酬（期初余额）] REAL,
    [员工人均工资（万）] REAL,
    员工工资占营业收入比率 REAL,
    营业收入 REAL,
    归属于上市公司股东的净利润 REAL,
    归属于上市公司股东的扣除非经常性损益的净利润 REAL,
    经营活动产生的现金流量净额 REAL,
    归属于上市公司股东的净资产 REAL,
    总资产 REAL,
    净资产比率 REAL,
    总资产利润率 REAL,
    现金分红金额（含税） REAL,
    现金分红金额占合并报表中归属于上市公司普通股股东的净利润的比率 REAL,
    --营业收入（上期） REAL,
    --归属于上市公司股东的净利润（上期） REAL,
    --归属于上市公司股东的扣除非经常性损益的净利润（上期） REAL,
    --经营活动产生的现金流量净额（上期） REAL,
    归属于上市公司股东的净资产（上期） REAL,
    总资产（上期） REAL,
    人均营业额（万） REAL,
    人均净利润（万） REAL,
    经营活动产生的现金流量净额占净利润的比率 REAL,
    营业收入增长率 REAL,
    归属于上市公司股东的净利润增长率 REAL,
    归属于上市公司股东的扣除非经常性损益的净利润增长率 REAL,
    经营活动产生的现金流量净额增长率 REAL,
    营业成本 REAL,
    投资收益 REAL,
    营业利润 REAL,
    利润总额 REAL,
    所得税费用 REAL,
    净利润 REAL,
    毛利润 REAL,
    销售费用 REAL,
    管理费用 REAL,
    财务费用 REAL,
    研发费用 REAL,
    费用化研发投入 REAL,
    资本化研发投入 REAL,
    费用总额（销售管理财务） REAL,
    资产减值准备 REAL,
    折旧和摊销费用 REAL,
    所得税费用占剔除投资收益后利润的比率 REAL,
    --三费总额占营业收入的比率 REAL,
    费用总额占营业收入的比率 REAL,
    资产减值和折旧摊销占营业收入的比率 REAL,
    营业利润率 REAL,
    毛利率 REAL,
    平均净资产收益率 REAL,
    净利率 REAL,
    营业收入占平均总资产的比率 REAL,
    平均总资产和净资产的比率 REAL,
    净资产增长率 REAL,
    研发投入占营业收入的比率 REAL,
    资本化研发投入的比率 REAL,
    --"无形资产-内部研发","所得税税率",
    总股本 REAL,
    每股净利润 REAL,
    每股净资产 REAL,
    固定资产 REAL,
    在建工程 REAL,
    在建工程占固定资产的比率 REAL,
    土地使用权 REAL,
    投资性房地产 REAL,
    商誉 REAL,
    生产资本 REAL,
    利润总额占生产资本的比率 REAL,
    商誉占营业收入的比率 REAL,
    预收款项 REAL,
    应付票据及应付账款 REAL,
    应付账款 REAL,
    预付款项 REAL,
    应收账款 REAL,
    应收票据 REAL,
    --应收票据（上期）REAL,
    应收账款占营业收入的比率 REAL,
    预收总额和应收总额的比率 REAL,
    预收款项和应收账款的比率 REAL,
    流动资产合计 REAL,
    负债合计 REAL,
    流动负债合计 REAL,
    存货 REAL,
    货币资金 REAL,
    短期借款 REAL,
    合同负债 REAL,
    一年内到期的非流动负债 REAL,--"有息债券(短期借款＋一年内到期的非流动负)","应收票据增加额（本期应收票据余额-上期应收票据余额）",
    短期借债总额 REAL,
    长期借款 REAL,
    应付债券 REAL,
    流动比率 REAL,
    速动比率 REAL,
    现金及现金等价物余额占短期借债的比率 REAL,
    流动资产占总负债的比率 REAL,
    销售商品、提供劳务收到的现金 REAL,
    现金及现金等价物余额 REAL,
    现金及现金等价物净增加额 REAL,
    现金收入和营业收入的比率 REAL,
    应收账款周转率 REAL,
    存货周转率 REAL,
    营业利润占营业资金的比率 REAL,
    营业收入占营业资金的比率 REAL,
    还原后的净资产收益率（ROCE） REAL
);

insert into {0}财务分析综合表
select
    a.报告时间,
    a.公司代码,
    a.报告类型,
    a.公司简称,
    a.公司名称,
    a.公司地址,
    a.行业分类,
    a.发布时间,
    0 as 公司投资等级,
    0 as 预测市值增长比值,
    a.在职员工的数量合计,
    a.支付给职工及为职工支付的现金,
    a.应付职工薪酬（期末余额）,
    a.应付职工薪酬（期初余额）,
    round((a.支付给职工及为职工支付的现金 + a.应付职工薪酬（期末余额） - a.应付职工薪酬（期初余额）)
        /replace(a.在职员工的数量合计,',','')/10000,2)
        as [员工人均工资（万）],
    round((a.支付给职工及为职工支付的现金 + a.应付职工薪酬（期末余额） - a.应付职工薪酬（期初余额）)/a.营业收入,4)
        as 员工工资占营业收入比率,
    a.营业收入,
    a.归属于上市公司股东的净利润,
    a.归属于上市公司股东的扣除非经常性损益的净利润,
    a.经营活动产生的现金流量净额,
    a.归属于上市公司股东的净资产,
    a.总资产,
    round(a.归属于上市公司股东的净资产/a.总资产,4) as 净资产比率,
    round(a.归属于上市公司股东的净利润/a.总资产,4) as 资本利润率,
    a.现金分红金额（含税）,
    a.现金分红金额占合并报表中归属于上市公司普通股股东的净利润的比率,
    a.归属于上市公司股东的净资产（上期）,
    a.总资产（上期）,
    round(a.营业收入/replace(a.在职员工的数量合计,',','')/10000,2) as [人均营业额（万）],
    round(a.归属于上市公司股东的净利润/replace(a.在职员工的数量合计,',','')/10000,2) as [人均净利润（万）],
    round(a.经营活动产生的现金流量净额/a.归属于上市公司股东的扣除非经常性损益的净利润,2) as 经营活动产生的现金流量净额占净利润的比率,
    round((a.营业收入 - a.营业收入（上期）)/abs(a.营业收入（上期）),4) as 营业收入增长率,   -- 分母需要加abs, 这样计算出来的增长率符号和分子是一致的
    round((a.归属于上市公司股东的净利润 - a.归属于上市公司股东的净利润（上期）)/abs(a.归属于上市公司股东的净利润（上期）),4)
        as 归属于上市公司股东的净利润增长率,
    case when a.归属于上市公司股东的扣除非经常性损益的净利润（上期） != '' and a.归属于上市公司股东的扣除非经常性损益的净利润（上期） is not NULL
        then round((a.归属于上市公司股东的扣除非经常性损益的净利润 - a.归属于上市公司股东的扣除非经常性损益的净利润（上期）)/abs(a.归属于上市公司股东的扣除非经常性损益的净利润（上期）),4)
        else 0 end
        as 归属于上市公司股东的扣除非经常性损益的净利润增长率,
    round((a.经营活动产生的现金流量净额 - a.经营活动产生的现金流量净额（上期）)/abs(a.经营活动产生的现金流量净额（上期）),4)
        as 经营活动产生的现金流量净额增长率,
    a.营业成本,
    a.投资收益,
    a.三、营业利润 as 营业利润,
    a.四、利润总额 as 利润总额,
    a.所得税费用,
    a.五、净利润 as 净利润,
    a.营业收入 - a.营业成本 as 毛利润,
    a.销售费用,
    a.管理费用,
    a.财务费用,
    a.研发费用,
    a.费用化研发投入,
    a.资本化研发投入,
    a.销售费用 + a.管理费用 + a.财务费用 + a.研发费用 as 费用总额（销售管理财务）,
    a.资产减值准备,
    a.固定资产折旧、油气资产折耗、生产性生物资产折旧 + a.无形资产摊销 + a.长期待摊费用摊销
        as 折旧和摊销费用,
    round(a.所得税费用/(a.四、利润总额 - a.投资收益),4) as 所得税费用占剔除投资收益后利润的比率,
    --无形资产-内部研发","资产减值准备","折旧和摊销","所得税税率",
    --round((a.销售费用 + a.管理费用 + a.财务费用)/a.营业收入,4)
    --    as 三费总额占营业收入的比率,
    round((a.销售费用 + a.管理费用 + a.财务费用 + a.研发费用)/a.营业收入,4)
        as 费用总额占营业收入的比率,
    round((a.资产减值准备 + a.固定资产折旧、油气资产折耗、生产性生物资产折旧 + a.无形资产摊销 + a.长期待摊费用摊销)/a.营业收入,4)
        as 资产减值和折旧摊销占营业收入的比率,
    round(a.三、营业利润/a.营业收入,4) as 营业利润率,
    round((a.营业收入 - a.营业成本)/a.营业收入,4) as 毛利率,
    round(a.五、净利润*2/(a.归属于上市公司股东的净资产 + a.归属于上市公司股东的净资产（上期）),4) as 平均净资产收益率,
    round(a.五、净利润/a.营业收入,4) as 净利率,
    round(a.营业收入*2/(a.总资产 + a.总资产（上期）),4) as 营业收入占平均总资产的比率,
    round((a.总资产 + a.总资产（上期）)/(a.归属于上市公司股东的净资产 + a.归属于上市公司股东的净资产（上期）),4) as 平均总资产和净资产的比率,
    round((a.归属于上市公司股东的净资产 - a.归属于上市公司股东的净资产（上期）)/a.归属于上市公司股东的净资产（上期）,4)
        as 净资产增长率,
    round((a.资本化研发投入 + a.费用化研发投入)/a.营业收入,4) as 研发投入占营业收入的比率,
    case when a.资本化研发投入 + a.费用化研发投入 != 0 then round(a.资本化研发投入/(a.资本化研发投入 + a.费用化研发投入),4) else 0 end
        as 资本化研发投入的比率,
    a.期末总股本 as 总股本,
    round(a.归属于上市公司股东的净利润/a.期末总股本,2) as 每股净利润,
    round(a.归属于上市公司股东的净资产/a.期末总股本,2) as 每股股净资产,
    a.固定资产,
    a.在建工程,
    round(a.在建工程/a.固定资产,4) as 在建工程占固定资产的比率,
    a.土地使用权,
    a.投资性房地产,
    a.商誉,
    a.固定资产 + a.在建工程 + a.土地使用权 + a.投资性房地产 as 生产资本,
    round(a.四、利润总额/(a.固定资产 + a.在建工程 + a.土地使用权 + a.投资性房地产),4)
        as 利润总额占生产资本的比率,
    round(a.商誉/a.营业收入,4) as 商誉占营业收入的比率,
    a.预收款项,
    a.应付票据及应付账款,
    a.应付账款,
    a.预付款项,
    a.应收账款,
    a.应收票据,
    round(a.应收账款/a.营业收入,2) as 应收账款占营业收入的比率,
    round((a.预收款项 + a.应付票据及应付账款)/(a.预付款项 + a.应收账款 + a.应收票据),2)
        as 预收总额和应收总额的比率,
    case when a.应收账款 != 0 then round(a.预收款项/a.应收账款,2) else 5000.0  end as 预收款项和应收账款的比率,
    a.流动资产合计,
    a.负债合计,
    a.流动负债合计,
    a.存货,
    a.货币资金,
    a.短期借款,
    a.合同负债,
    a.一年内到期的非流动负债,--"有息债券(短期借款＋一年内到期的非流动负)","应收票据增加额（本期应收票据余额-上期应收票据余额）",
    a.短期借款 + a.一年内到期的非流动负债 as 短期借债总额,
    a.长期借款,
    a.应付债券,
    round(a.流动资产合计/a.流动负债合计,2) as 流动比率,
    round((a.流动资产合计 - a.存货)/a.流动负债合计,2) as 速动比率,
    case when (a.短期借款 + a.一年内到期的非流动负债) = 0 then 100.0
        else round(a.六、期末现金及现金等价物余额/(a.短期借款 + a.一年内到期的非流动负债),2)
        end as 现金及现金等价物余额占短期借债的比率,
    round(a.流动资产合计/a.负债合计,2) as 流动资产占总负债的比率,
    a.销售商品、提供劳务收到的现金,
    a.[六、期末现金及现金等价物余额] as 现金及现金等价物余额,
    a.五、现金及现金等价物净增加额 as 现金及现金等价物净增加额,
    round(a.销售商品、提供劳务收到的现金/a.营业收入,2) 现金收入和营业收入的比率,
    case when a.应收账款 != 0 then round(a.营业收入/a.应收账款,2)  else 10000.0 end as 应收账款周转率,
    case when a.存货 != 0 then round(a.营业成本/a.存货,2) else 100000.0 end
        as 存货周转率,
    round(a.三、营业利润/(a.流动资产合计 - a.流动负债合计),2) as 营业利润占营业资金的比率,
    round(a.营业收入/(a.流动资产合计 - a.流动负债合计),2) as 营业收入占营业资金的比率,
    round(a.归属于上市公司股东的净利润/a.归属于上市公司股东的净资产（上期）,2) 还原后的净资产收益率（ROCE）
from {0}财务分析基础表 a;

CREATE INDEX IF NOT EXISTS [{0}财务分析综合表索引] on [{0}财务分析综合表] (
    报告时间,
    公司代码,
    报告类型
);


