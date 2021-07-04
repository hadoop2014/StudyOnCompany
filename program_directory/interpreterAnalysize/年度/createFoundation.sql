--参数{0}会被替换成报告类型,如: 年报,半年报,季报
drop table if exists {0}财务分析基础表;
create table if not exists {0}财务分析基础表 (
    --ID INTEGER PRIMARY KEY AUTOINCREMENT,
    报告时间 DATE NOT NULL,
    公司代码 INTEGER NOT NULL,
    报告类型 CHAR(20) NOT NULL,
    公司简称 CHAR(10),
    公司名称 CHAR(50),
    公司地址 CHAR(10),
    行业分类 CHAR(10),
    发布时间 DATE,
    在职员工的数量合计 REAL,
    支付给职工及为职工支付的现金 REAL,
    应付职工薪酬（期末余额） REAL,
    应付职工薪酬（期初余额） REAL,
    营业收入 REAL,
    归属于上市公司股东的净利润 REAL,
    归属于上市公司股东的扣除非经常性损益的净利润 REAL,
    经营活动产生的现金流量净额 REAL,
    归属于上市公司股东的净资产 REAL,
    总资产 REAL,
    现金分红金额（含税） REAL,
    现金分红金额占合并报表中归属于上市公司普通股股东的净利润的比率 REAL,
    营业收入（上期） REAL,
    归属于上市公司股东的净利润（上期） REAL,
    归属于上市公司股东的扣除非经常性损益的净利润（上期） REAL,
    经营活动产生的现金流量净额（上期） REAL,
    归属于上市公司股东的净资产（上期） REAL,
    总资产（上期） REAL,
    营业成本 REAL,
    投资收益 REAL,
    三、营业利润 REAL,
    四、利润总额 REAL,
    所得税费用 REAL,
    五、净利润 REAL,
    销售费用 REAL,
    管理费用 REAL,
    财务费用 REAL,
    研发费用 REAL,
    资本化研发投入 REAL,
    资产减值准备 REAL,
    固定资产折旧、油气资产折耗、生产性生物资产折旧 REAL,
    无形资产摊销 REAL,
    长期待摊费用摊销 REAL,
    --"无形资产-内部研发","所得税税率",
    期末总股本 REAL,
    固定资产 REAL,
    在建工程 REAL,
    土地使用权 REAL,
    投资性房地产 REAL,
    商誉 REAL,
    预收款项 REAL,
    应付票据及应付账款 REAL,
    应付账款 REAL,
    预付款项 REAL,
    应收账款 REAL,
    应收票据 NMERIC,
    流动资产合计 REAL,
    负债合计 REAL,
    流动负债合计 REAL,
    存货 REAL,
    货币资金 REAL,
    短期借款 REAL,
    一年内到期的非流动负债 REAL,
    长期借款 REAL,
    应付债券 REAL,
    销售商品、提供劳务收到的现金 REAL,
    六、期末现金及现金等价物余额 REAL,
    五、现金及现金等价物净增加额 REAL
);

insert into {0}财务分析基础表
select
    a.报告时间,
    a.公司代码,
    a.报告类型,
    a.公司简称,
    a.公司名称,
    a.公司地址,
    case when a.行业分类 != '' then a.行业分类 else i.行业分类 end as 行业分类,
    j.发布时间,
    case when a.在职员工的数量合计 != '' then a.在职员工的数量合计 else a.当期领取薪酬员工总人数 end as 在职员工的数量合计,
    f.支付给职工及为职工支付的现金,
    case when c.应付职工薪酬 != '' then c.货币单位 * replace(c.应付职工薪酬,',','') else 0 end as 应付职工薪酬（期末余额）,
    c.应付职工薪酬（期初余额）,
    b.营业收入,
    b.归属于上市公司股东的净利润,
    b.归属于上市公司股东的扣除非经常性损益的净利润,
    case when b.经营活动产生的现金流量净额 is not NULL then b.经营活动产生的现金流量净额 else f.经营活动产生的现金流量净额 end
        as 经营活动产生的现金流量净额,
    b.归属于上市公司股东的净资产,
    b.总资产,
    d.现金分红金额（含税）,
    --round(replace(d.现金分红金额占合并报表中归属于上市公司普通股股东的净利润的比率,'%','')/100,4)
    --    as 现金分红金额占合并报表中归属于上市公司普通股股东的净利润的比率,
    d.现金分红金额占合并报表中归属于上市公司普通股股东的净利润的比率,
    b.营业收入（上期）,
    b.归属于上市公司股东的净利润（上期）,
    b.归属于上市公司股东的扣除非经常性损益的净利润（上期）,
    case when b.经营活动产生的现金流量净额（上期） is not NULL then b.经营活动产生的现金流量净额（上期） else f.经营活动产生的现金流量净额（上期） end
        as 经营活动产生的现金流量净额（上期）,
    case when b.归属于上市公司股东的净资产（上期） is not NULL then b.归属于上市公司股东的净资产（上期） else c.归属于上市公司股东的净资产（上期） end
        as 归属于上市公司股东的净资产（上期）,
    case when b.总资产（上期） is not NULL then b.总资产（上期） else c.总资产（上期） end
        as 总资产（上期）,
    e.货币单位 * replace(e.营业成本,',','') as 营业成本,
    case when e.投资收益 != '' then e.货币单位 * replace(e.投资收益,',','') else 0 end as 投资收益,
    e.货币单位 * replace(e.三、营业利润,',','') as 三、营业利润,
    e.货币单位 * replace(e.四、利润总额,',','') as 四、利润总额,
    e.货币单位 * replace(e.所得税费用,',','') as 所得税费用,
    e.货币单位 * replace(e.五、净利润,',','') as 五、净利润,
    case when e.销售费用 is not NULL and e.销售费用 != '' then e.货币单位 * replace(e.销售费用,',','') else 0 end
        as 销售费用,
    case when e.管理费用 is not NULL and e.管理费用 != '' then e.货币单位 * replace(e.管理费用,',','') else 0 end
        as 管理费用,
    case when e.财务费用 is not NULL and e.财务费用 != '' then e.货币单位 * replace(e.财务费用,',','') else 0 end
        as 财务费用,
    case when a.本期费用化研发投入修正 != '' then a.本期费用化研发投入修正 else
        case when e.研发费用 is not NULL and e.研发费用 != '' then e.货币单位 * replace(e.研发费用,',','') else 0 end
        end as 研发费用,
    case when a.本期资本化研发投入修正 != '' then a.本期资本化研发投入修正 else 0 end as 资本化研发投入,
    case when g.资产减值准备 != '' then g.货币单位 * replace(g.资产减值准备,',','') else 0 end as 资产减值准备 ,
    case when g.固定资产折旧、油气资产折耗、生产性生物资产折旧 is not NULL and g.固定资产折旧、油气资产折耗、生产性生物资产折旧 != ''
    then
        case when g.使用权资产摊销 is not NULL and g.使用权资产摊销 != ''
        then g.货币单位 * replace(g.固定资产折旧、油气资产折耗、生产性生物资产折旧,',','') + g.货币单位 * replace(g.使用权资产摊销,',','')
        else g.货币单位 * replace(g.固定资产折旧、油气资产折耗、生产性生物资产折旧,',','')  end
    else
        case when g.使用权资产摊销 is not NULL and g.使用权资产摊销 != ''
        then g.货币单位 * replace(g.使用权资产摊销,',','')
        else 0 end
    end as 固定资产折旧、油气资产折耗、生产性生物资产折旧,
    case when g.无形资产摊销 is not NULL then g.货币单位 * replace(g.无形资产摊销,',','') else 0 end as 无形资产摊销,
    case when g.长期待摊费用摊销 is not NULL then g.货币单位 * replace(g.长期待摊费用摊销,',','') else 0 end as 长期待摊费用摊销,
    --"无形资产-内部研发","所得税税率",
    b.期末总股本,
    c.货币单位 * replace(c.固定资产,',','') as 固定资产,
    case when c.在建工程 != '' then c.货币单位 * replace(c.在建工程,',','') else 0 end as 在建工程,
    case when h.期末账面价值 != '' then h.期末账面价值 else 0 end as 土地使用权,
    case when c.投资性房地产 is not NULL and c.投资性房地产 != '' then c.货币单位 * replace(c.投资性房地产,',','') else 0 end as 投资性房地产,
    --case when c.商誉 != '' then c.商誉 else 0 end as 商誉,
    case when c.商誉 != '' then c.货币单位 * replace(c.商誉,',','') else 0 end as 商誉,
    case when c.预收款项 is not NULL and 预收款项 != '' then c.货币单位 * replace(c.预收款项,',','') else 0 end as 预收款项,
    case when c.应付票据及应付账款 is not NULL then c.货币单位 * replace(c.应付票据及应付账款,',','') else
        case when c.应付账款 is not NULL and c.货币单位 * replace(c.应付票据,',','') is not NULL
            then c.货币单位 * replace(c.应付账款,',','') + c.货币单位 * replace(c.应付票据,',','') else 0
        end
    end as 应付票据及应付账款,
    case when c.应付账款 is not NULL then c.货币单位 * replace(c.应付账款,',','') else 0 end as 应付账款,
    case when c.预付款项 is not NULL then c.货币单位 * replace(c.预付款项,',','') else 0 end as 预付款项,
    case when c.应收票据及应收账款 is not NULL and c.应收账款 is NULL then c.货币单位 * replace(c.应收票据及应收账款,',','') else
        case when c.应收账款 is not NULL and c.应收账款 != '' then c.货币单位 * replace(c.应收账款,',','') else 0 end
    end as 应收账款,
    case when c.应收票据 is not NULL and c.应收票据 != '' then c.货币单位 * replace(c.应收票据,',','') else 0 end as 应收票据,
    case when c.流动资产合计 is not NULL then c.货币单位 * replace(c.流动资产合计,',','')
        -- 解决国金证券 没有 流动资产合计 字段
        else c.货币单位 * replace(c.资产总计,',','')
            - case when c.可供出售金融资产 != '' then c.货币单位 * replace(c.可供出售金融资产,',','') else 0 end
            - case when c.持有至到期投资 != '' then c.货币单位 * replace(c.持有至到期投资,',','') else 0 end
            - case when c.长期股权投资 != '' then c.货币单位 * replace(c.长期股权投资,',','') else 0 end
            - case when c.投资性房地产 != '' then c.货币单位 * replace(c.投资性房地产,',','') else 0 end
            - case when c.固定资产 != '' then c.货币单位 * replace(c.固定资产,',','') else 0 end
            - case when c.在建工程 != '' then c.货币单位 * replace(c.在建工程,',','') else 0 end
            - case when c.无形资产 != '' then c.货币单位 * replace(c.无形资产,',','') else 0 end
            - case when c.商誉 != '' then c.货币单位 * replace(c.商誉,',','') else 0 end
            - case when c.长期待摊费用 != '' and c.长期待摊费用 is not NULL then c.货币单位 * replace(c.长期待摊费用,',','') else 0 end
            - case when c.递延所得税资产 != '' then c.货币单位 * replace(c.递延所得税资产,',','') else 0 end
        end
        as 流动资产合计,
    c.货币单位 * replace(c.负债合计,',','') as 负债合计,
    case when c.流动负债合计 is not NULL then c.货币单位 * replace(c.流动负债合计,',','')
        -- 解决国金证券 没有 流动负债合计 字段
        else c.货币单位 * replace(c.负债合计,',','')
            - case when c.长期借款 != '' then c.货币单位 * replace(c.长期借款,',','') else 0 end
            - case when c.应付债券 != '' then c.货币单位 * replace(c.应付债券,',','') else 0 end
            - case when c.递延所得税负债 != '' then c.货币单位 * replace(c.递延所得税负债,',','') else 0 end
        --    - iif(c.长期借款 != '', replace(c.长期借款,',',''),0)
        --    - iif(c.应付债券 != '', replace(c.应付债券,',',''),0)
        --    - iif(c.递延所得税负债 != '' , replace(c.递延所得税负债,',',''),0)
        end
        as 流动负债合计,
    case when c.存货 != '' then c.货币单位 * replace(c.存货,',','') else 0 end as 存货,
    c.货币单位 * replace(c.货币资金,',','') as 货币单位,
    case when c.短期借款 is not NULL and c.短期借款 != '' then c.货币单位 * replace(c.短期借款,',','') else 0 end as 短期借款,
    case when c.一年内到期的非流动负债 is not NULL and c.一年内到期的非流动负债 != ''
        then c.货币单位 * replace(c.一年内到期的非流动负债,',','') else 0 end
        as 一年内到期的非流动负债,
    case when c.长期借款 is not NULL and c.长期借款 != '' then c.货币单位 * replace(c.长期借款,',','') else 0 end as 长期借款,
    case when c.应付债券 is not NULL and c.应付债券 != '' then c.货币单位 * replace(c.应付债券,',','') else 0 end as 应付债券,
    case when f.销售商品、提供劳务收到的现金 is not NULL then f.销售商品、提供劳务收到的现金
        -- 解决国金证券 没有 销售商品、提供劳务收到的现金 字段, 用 收取利息、手续费及佣金的现金 字段来取代
        else f.收取利息、手续费及佣金的现金 end
        as 销售商品、提供劳务收到的现金,
    f.六、期末现金及现金等价物余额,
    f.五、现金及现金等价物净增加额
from
(
    select x.*,
        case when 本期费用化研发投入 = '' and 研发投入金额 != '' and 本期资本化研发投入 != ''
            then x.货币单位 * round(replace(研发投入金额,',','') - x.货币单位 * replace(本期资本化研发投入,',',''),2)
            else x.货币单位 * replace(本期费用化研发投入,',','') end
            as 本期费用化研发投入修正,
        case when 本期资本化研发投入 = '' and 研发投入金额 != '' and 本期费用化研发投入 != ''
            then x.货币单位 * round(replace(研发投入金额,',','') - x.货币单位 * replace(本期费用化研发投入,',',''),2)
            else x.货币单位 * replace(本期资本化研发投入,',','') end
            as 本期资本化研发投入修正
    from {0}关键数据表 x
    where (x.在职员工的数量合计 != '' or x.当期领取薪酬员工总人数 != '') and x.报告类型 = '{0}报告'
)a
left join
(
    select x.报告时间,
        x.公司代码,
        x.公司简称,
        x.报告类型,
        x.货币单位 * replace(x.营业收入,',','') as 营业收入,
        x.货币单位 * replace(x.归属于上市公司股东的净利润,',','') as 归属于上市公司股东的净利润,
        x.货币单位 * replace(x.归属于上市公司股东的扣除非经常性损益的净利润,',','') as 归属于上市公司股东的扣除非经常性损益的净利润,
        x.货币单位 * replace(x.经营活动产生的现金流量净额,',','') as 经营活动产生的现金流量净额,
        case when x.归属于上市公司股东的净资产 is not NULL
            then x.货币单位 * replace(x.归属于上市公司股东的净资产,',','')
            else z.货币单位 * replace(z.归属于母公司所有者权益（或股东权益）合计,',','')  end as 归属于上市公司股东的净资产,
        case when x.总资产 is not NULL
            then x.货币单位 * replace(x.总资产,',','')
            else z.货币单位 * replace(z.负债和所有者权益（或股东权益）总计,',','') end as 总资产,
        case when x.期末总股本 is not NULL
            then x.货币单位 * replace(x.期末总股本,',','')
            else z.货币单位 * replace(z.实收资本（或股本）,',','') end as 期末总股本,
        y.货币单位 * replace(y.营业收入,',','') as 营业收入（上期）,
        y.货币单位 * replace(y.归属于上市公司股东的净利润,',','') as 归属于上市公司股东的净利润（上期）,
        y.货币单位 * replace(y.归属于上市公司股东的扣除非经常性损益的净利润,',','') as 归属于上市公司股东的扣除非经常性损益的净利润（上期）,
        y.货币单位 * replace(y.经营活动产生的现金流量净额,',','') as 经营活动产生的现金流量净额（上期）,
        y.货币单位 * replace(y.归属于上市公司股东的净资产,',','') as 归属于上市公司股东的净资产（上期）,
        y.货币单位 * replace(y.总资产,',','') as 总资产（上期）
    from {0}主要会计数据 x
    left join {0}主要会计数据 y
    left join {0}合并资产负债表 z
    where (x.报告时间 - y.报告时间 = 1 and x.报告类型 = y.报告类型 and x.公司代码 = y.公司代码)
        and (x.报告时间 = z.报告时间 and x.公司代码 = z.公司代码 and x.报告类型 = z.报告类型)
)b
left join
(
    select x.*,
        case when y.应付职工薪酬 != '' then y.货币单位 * replace(y.应付职工薪酬,',','')  else 0 end as 应付职工薪酬（期初余额）,
        y.货币单位 * replace(y.归属于母公司所有者权益（或股东权益）合计,',','')  as 归属于上市公司股东的净资产（上期）,
        y.货币单位 * replace(y.负债和所有者权益（或股东权益）总计,',','') as 总资产（上期）
    from {0}合并资产负债表 x
    left join {0}合并资产负债表 y
    where x.报告时间 - y.报告时间 = 1
        and x.报告类型 = y.报告类型
        and x.公司代码 = y.公司代码
)c
left join
(
    select x.报告时间,x.公司代码,x.报告类型,
        case when x.现金分红金额（含税） != ''
        then x.货币单位 * replace(x.现金分红金额（含税）,',','')
        else 0 end
            as 现金分红金额（含税）,
        case when x.现金分红金额占合并报表中归属于上市公司普通股股东的净利润的比率 != ''
        then
            round(replace(x.现金分红金额占合并报表中归属于上市公司普通股股东的净利润的比率,'%','')/100.0,4)
        else 0 end as 现金分红金额占合并报表中归属于上市公司普通股股东的净利润的比率
    from 年度普通股现金分红情况表 x
)d
left join {0}合并利润表 e
left join (
    select x.报告时间,x.报告类型,x.公司代码,x.公司简称,
        x.货币单位 * replace(x.支付给职工及为职工支付的现金,',','')  as 支付给职工及为职工支付的现金,
        x.货币单位 * replace(x.销售商品、提供劳务收到的现金,',','')  as 销售商品、提供劳务收到的现金,
        x.货币单位 * replace(x.收取利息、手续费及佣金的现金,',','')  as 收取利息、手续费及佣金的现金,
        x.货币单位 * replace(x.六、期末现金及现金等价物余额,',','')  as 六、期末现金及现金等价物余额,
        x.货币单位 * replace(x.五、现金及现金等价物净增加额,',','')  as 五、现金及现金等价物净增加额,
        x.货币单位 * replace(x.经营活动产生的现金流量净额,',','')  as 经营活动产生的现金流量净额,
        y.货币单位 * replace(y.经营活动产生的现金流量净额,',','')  as 经营活动产生的现金流量净额（上期）
    from {0}合并现金流量表 x
    left join {0}合并现金流量表 y
    on (x.报告时间 - y.报告时间 = 1 and x.报告类型 = y.报告类型 and x.公司代码 = y.公司代码)
)f
left join {0}现金流量表补充资料 g
left join
(
    --针对有些无形资产情况中,没有土地使用权的情况,如东方财富2017,2018年报,默认设置 期末账面价值 为0
    select x.报告时间,x.公司代码,x.报告类型,x.公司简称,
        case when y.期末账面价值 is not NULL and y.期末账面价值 != '' then y.期末账面价值 else x.期末账面价值 end as 期末账面价值
    from
    (
        select 报告时间,公司代码,报告类型,公司简称,0 as 期末账面价值
        from {0}无形资产情况
        group by 报告时间,公司代码,报告类型
    )x
    left join
    (
        select 报告时间,公司代码,报告类型,公司简称,货币单位 * replace(期末账面价值,',','') as 期末账面价值
        from {0}无形资产情况 x
        where 项目 = '土地使用权'
    )y
    on x.报告时间 = y.报告时间 and x.公司代码 = y.公司代码 and x.报告类型 = y.报告类型
)h
left join 行业分类数据 i
on a.公司代码 = i.公司代码
left join
(
    --同一个公司同一年度可能会发布两次财报, 第二次为第一次发布财报的修正, 但是我们认为第一次发布的财报影响更大, 因此取第一次发布的时间为准
    select 公司代码, 报告时间, 报告类型, min(发布时间) as 发布时间
    from 财报发布信息
    group by 公司代码, 报告时间, 报告类型
)j
where (a.报告时间 = b.报告时间 and a.公司代码 = b.公司代码 and a.报告类型 = b.报告类型)
    and (a.报告时间 = c.报告时间 and a.公司代码 = c.公司代码 and a.报告类型 = c.报告类型)
    and (a.报告时间 = d.报告时间 and a.公司代码 = d.公司代码 and a.报告类型 = d.报告类型)
    and (a.报告时间 = e.报告时间 and a.公司代码 = e.公司代码 and a.报告类型 = e.报告类型)
    and (a.报告时间 = f.报告时间 and a.公司代码 = f.公司代码 and a.报告类型 = f.报告类型)
    and (a.报告时间 = g.报告时间 and a.公司代码 = g.公司代码 and a.报告类型 = g.报告类型)
    and (a.报告时间 = h.报告时间 and a.公司代码 = h.公司代码 and a.报告类型 = h.报告类型)
    --and (a.公司代码 = i.公司代码)
    and (a.报告时间 = j.报告时间 and a.公司代码 = j.公司代码 and a.报告类型 = j.报告类型)
    and (a.报告时间 != '' and a.公司代码 != '' and a.报告类型 != '')
order by a.报告时间,a.公司代码,a.报告类型;

CREATE INDEX IF NOT EXISTS [{0}财务分析基础表索引] on [{0}财务分析基础表] (
    报告时间,
    公司代码,
    报告类型
);


