--参数{0}会被替换成报告类型,如: 年报,半年报,季报
drop table if exists {0}公司价格分析中间表;
create table if not exists {0}公司价格分析中间表 (
    报告时间 DATE NOT NULL,
    公司代码 INTEGER NOT NULL,
    报告类型 CHAR(20) NOT NULL,
    公司简称 CHAR(10),
    发布时间 DATE,
    结束时间 DATE,
    间隔时长 REAL,
    文件名 CHAR(20),
    报告周 CHAR(10),
    报告周总市值 REAL,
    结束周 CHAR(10),
    结束周总市值 REAL,
    起始周 CHAR(10),
    起始周总市值 REAL,
    起始周指数 REAL,
    报告周指数 REAL,
    本年市值增长率 REAL,
    市值增长率 REAL,
    本年指数增长率 REAL,
    报告周上证指数 REAL,
    结束周上证指数 REAL,
    上证指数增长率 REAL,
    深证成指增长率 REAL,
    创业板指增长率 REAL,
    沪深300指数增长率 REAL,
    训练标识 INTEGER DEFAULT NULL
);


insert into {0}公司价格分析中间表
select
    a.报告时间,
    a.公司代码,
    a.报告类型,
    a.公司简称,
    a.发布时间,
    a.结束时间,
    round((julianday(a.结束时间) - julianday(a.发布时间))/365.0,2) as 间隔时长,
    a.文件名,
    a.报告周,
    b.周平均总市值 as 报告周总市值,
    a.结束周,
    c.周平均总市值 as 结束周总市值,
    a.起始周,
    o.周平均总市值 as 起始周总市值,
    e.周平均收盘价 as 起始周指数,
    d.周平均收盘价 as 报告周指数,
    round((b.周平均总市值 - o.周平均总市值)/o.周平均总市值, 4) as 本年市值增长率,
    round((c.周平均总市值 - b.周平均总市值)/b.周平均总市值, 4) as 市值增长率,
    --case when a.公司代码 < '300000' then round((f.周平均收盘价 - g.周平均收盘价)/g.周平均收盘价, 4) else
    --    case when a.公司代码 < '600000' then round((j.周平均收盘价 - k.周平均收盘价)/k.周平均收盘价, 4) else
    --        round((h.周平均收盘价 - i.周平均收盘价)/i.周平均收盘价, 4)
    --    end
    --end as 本年指数增长率,
    round((d.周平均收盘价 - e.周平均收盘价)/e.周平均收盘价, 4) as 本年指数增长率,
    f.周平均收盘价 as 报告周上证指数,
    g.周平均收盘价 as 起始周上证指数,
    round((f.周平均收盘价 - g.周平均收盘价)/g.周平均收盘价, 4) as 上证指数增长率,
    round((h.周平均收盘价 - i.周平均收盘价)/i.周平均收盘价, 4) as 深证成指增长率,
    round((j.周平均收盘价 - k.周平均收盘价)/k.周平均收盘价, 4) as 创业板指增长率,
    round((d.周平均收盘价 - e.周平均收盘价)/e.周平均收盘价, 4) as 沪深300指数增长率,
    a.训练标识
from
(
    select x.报告时间,
        x.公司代码,
        x.公司简称,
        x.报告类型,
        x.发布时间,
        x.文件名,
        strftime('%Y-%W', x.发布时间) as 报告周,
        case when x.结束时间 is not NULL
            then strftime('%Y-%W', x.结束时间)
            else
                case when strftime('%Y-%W',x.发布时间, '+1 year') > z.结束周
                    then z.结束周
                    else strftime('%Y-%W',x.发布时间, '+1 year')
                    end
            end
            as 结束周,
        case when x.结束时间 is not NULL
            then x.结束时间
            else
                case when strftime('%Y-%W',x.发布时间, '+1 year') > z.结束周
                    then z.结束时间
                    else date(x.发布时间, '+1 year')
                    end
            end
            as 结束时间,
        case when x.起始时间 is not NULL
            then strftime('%Y-%W', x.起始时间)
            else
                case when strftime('%Y-%W',x.发布时间, '-1 year') < z.起始周
                    then z.起始周
                    else strftime('%Y-%W',x.发布时间, '-1 year')
                    end
            end
            as 起始周,
        case when x.起始时间 is not NULL
            then x.起始时间
            else
                case when strftime('%Y-%W',x.发布时间, '-1 year') > z.起始周
                    then z.起始时间
                    else date(x.发布时间, '-1 year')
                    end
            end
            as 起始时间,
        case when x.结束时间 is not NULL then 1 else 0 end as 训练标识

        --case when strftime('%Y-%W',x.发布时间, '+1 year') > z.结束周
        --    then z.结束周
        --    else strftime('%Y-%W',x.发布时间, '+1 year') end
        --    as 结束周,

        --strftime('%Y-%W',x.发布时间, '+1 year') as 最大周
        -- row_number() OVER(
        --    PARTITION BY x.公司代码, x.报告时间 ORDER BY x.发布时间  -- RANGER BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
        -- ) AS 去重标识
    from
    (
        select bb.*, aa.结束时间, aa.起始时间
        from
        (
            select ii.*,jj.发布时间 as 结束时间, kk.发布时间 as 起始时间
            from
            (
                -- 同一个公司同一年度可能会发布两次财报, 第二次为第一次发布财报的修正, 但是我们认为第一次发布的财报影响更大, 因此取第一次发布的时间为准
                select 公司代码, 报告时间, 报告类型, min(发布时间) as 发布时间
                from 财报发布信息
                group by 公司代码, 报告时间, 报告类型
            )ii
            left join
            (
                -- 同一个公司同一年度可能会发布两次财报, 第二次为第一次发布财报的修正, 但是我们认为第一次发布的财报影响更大, 因此取第一次发布的时间为准
                select 公司代码, 报告时间, 报告类型, min(发布时间) as 发布时间
                from 财报发布信息
                group by 公司代码, 报告时间, 报告类型
            )jj
            on ii.公司代码 = jj.公司代码 and ii.报告类型 = jj.报告类型 and jj.报告时间 - ii.报告时间 = 1
            left join
            (
                -- 同一个公司同一年度可能会发布两次财报, 第二次为第一次发布财报的修正, 但是我们认为第一次发布的财报影响更大, 因此取第一次发布的时间为准
                select 公司代码, 报告时间, 报告类型, min(发布时间) as 发布时间
                from 财报发布信息
                group by 公司代码, 报告时间, 报告类型
            )kk
            on ii.公司代码 = kk.公司代码 and ii.报告类型 = kk.报告类型 and ii.报告时间 - kk.报告时间 = 1
        )aa
        left join 财报发布信息 bb
        on aa.公司代码 = bb.公司代码 and aa.报告时间 = bb.报告时间 and aa.报告类型 = bb.报告类型 and aa.发布时间 = bb.发布时间
    )x
    left join
    (
        select 公司代码,
            max(报告时间) as 结束时间,
            max(报告周) as 结束周,
            min(报告时间) as 起始时间,
            min(报告周) as 起始周
        from
        (
            select 公司代码,
                报告时间,
                strftime('%Y-%W', 报告时间) as 报告周
            from 股票交易数据
            where 总市值 > 0 and 总市值 != 'None'   -- 解决立方制药 003020 出现无效数据,这些数据的总市值 = 0 or 总市值 = 'None'
        )y
        group by 公司代码
    )z
    where x.公司代码 = z.公司代码 and x.报告类型 = '{0}报告'
)a
left join
(
    select x.报告周,
        x.公司代码,
        round(avg(x.总市值), 0) as 周平均总市值
    from
    (
        select *,
            strftime('%Y-%W', 报告时间) as 报告周
        from 股票交易数据
    )x
    group by 报告周, 公司代码
)b
on a.公司代码 = b.公司代码 and a.报告周 = b.报告周
left join
(
    select x.报告周,
        x.公司代码,
        round(avg(x.总市值), 0) as 周平均总市值
    from
    (
        select *,
            strftime('%Y-%W', 报告时间) as 报告周
        from 股票交易数据
    )x
    group by 报告周, 公司代码
)c
on a.公司代码 = c.公司代码 and a.结束周 = c.报告周
left join
(
    select x.报告周,
        x.公司代码,
        round(avg(x.总市值), 0) as 周平均总市值
    from
    (
        select *,
            strftime('%Y-%W', 报告时间) as 报告周
        from 股票交易数据
    )x
    group by 报告周, 公司代码
)o
on a.公司代码 = o.公司代码 and a.起始周 = o.报告周
left join
(
    select x.报告周,
        x.公司代码,
        round(avg(x.收盘价), 0) as 周平均收盘价
    from
    (
        select *,
            strftime('%Y-%W', 报告时间) as 报告周
        from 股票交易数据
    )x
    where x.公司简称 = '沪深300'
    group by 报告周, 公司代码
)d
on a.报告周 = d.报告周
left join
(
    select x.报告周,
        x.公司代码,
        round(avg(x.收盘价), 0) as 周平均收盘价
    from
    (
        select *,
            strftime('%Y-%W', 报告时间) as 报告周
        from 股票交易数据
    )x
    where x.公司简称 = '沪深300'
    group by 报告周, 公司代码
)e
on a.起始周 = e.报告周
left join
(
    select x.报告周,
        x.公司代码,
        round(avg(x.收盘价), 0) as 周平均收盘价
    from
    (
        select *,
            strftime('%Y-%W', 报告时间) as 报告周
        from 股票交易数据
    )x
    where x.公司简称 = '上证指数'
    group by 报告周, 公司代码
)f
on a.报告周 = f.报告周
left join
(
    select x.报告周,
        x.公司代码,
        round(avg(x.收盘价), 0) as 周平均收盘价
    from
    (
        select *,
            strftime('%Y-%W', 报告时间) as 报告周
        from 股票交易数据
    )x
    where x.公司简称 = '上证指数'
    group by 报告周, 公司代码
)g
on a.起始周 = g.报告周
left join
(
    select x.报告周,
        x.公司代码,
        round(avg(x.收盘价), 0) as 周平均收盘价
    from
    (
        select *,
            strftime('%Y-%W', 报告时间) as 报告周
        from 股票交易数据
    )x
    where x.公司简称 = '深证成指'
    group by 报告周, 公司代码
)h
on a.报告周 = h.报告周
left join
(
    select x.报告周,
        x.公司代码,
        round(avg(x.收盘价), 0) as 周平均收盘价
    from
    (
        select *,
            strftime('%Y-%W', 报告时间) as 报告周
        from 股票交易数据
    )x
    where x.公司简称 = '深证成指'
    group by 报告周, 公司代码
)i
on a.起始周 = i.报告周
left join
(
    select x.报告周,
        x.公司代码,
        round(avg(x.收盘价), 0) as 周平均收盘价
    from
    (
        select *,
            strftime('%Y-%W', 报告时间) as 报告周
        from 股票交易数据
    )x
    where x.公司简称 = '创业板指'
    group by 报告周, 公司代码
)j
on a.报告周 = j.报告周
left join
(
    select x.报告周,
        x.公司代码,
        round(avg(x.收盘价), 0) as 周平均收盘价
    from
    (
        select *,
            strftime('%Y-%W', 报告时间) as 报告周
        from 股票交易数据
    )x
    where x.公司简称 = '创业板指'
    group by 报告周, 公司代码
)k
on  a.起始周 = k.报告周;


CREATE INDEX IF NOT EXISTS [{0}公司价格分析中间表索引] on [{0}公司价格分析中间表] (
    报告时间,
    公司代码,
    报告类型
);


drop table if exists {0}公司价格分析表;
create table if not exists {0}公司价格分析表 (
    报告时间 DATE NOT NULL,
    公司代码 INTEGER NOT NULL,
    报告类型 CHAR(20) NOT NULL,
    公司简称 CHAR(10),
    发布时间 DATE,
    结束时间 DATE,
    起始周 CHAR(10),
    报告周 CHAR(10),
    结束周 CHAR(10),
    起始周总市值 REAL,
    报告周总市值 REAL,
    结束周总市值 REAL,
    起始周指数 REAL,
    报告周指数 REAL,
    营业收入 REAL,
    归属于上市公司股东的净利润 REAL,
    归属于上市公司股东的扣除非经常性损益的净利润 REAL,
    -------经营活动产生的现金流量净额 REAL,
    员工工资占营业收入比率 REAL,
    净资产比率 REAL,
    总资产利润率 REAL,
    现金分红金额占合并报表中归属于上市公司普通股股东的净利润的比率 REAL,
    经营活动产生的现金流量净额占净利润的比率 REAL,
    ----营业收入增长率 REAL,
    ----归属于上市公司股东的净利润增长率 REAL,
    ----归属于上市公司股东的扣除非经常性损益的净利润增长率 REAL,
    ----经营活动产生的现金流量净额增长率 REAL,
    -----所得税费用占剔除投资收益后利润的比率 REAL,
    --三费总额占营业收入的比率 REAL,
    费用总额占营业收入的比率 REAL,
    资产减值和折旧摊销占营业收入的比率 REAL,
    营业利润率 REAL,
    毛利率 REAL,
    净利率 REAL,
    平均净资产收益率 REAL,
    营业收入占平均总资产的比率 REAL,
    净资产增长率 REAL,
    研发投入占营业收入的比率 REAL,
    资本化研发投入的比率 REAL,
    在建工程占固定资产的比率 REAL,
    利润总额占生产资本的比率 REAL,
    商誉占营业收入的比率 REAL,
    应收账款占营业收入的比率 REAL,
    --预收总额和应收总额的比率 REAL,
    --预收款项和应收账款的比率 REAL,
    流动比率 REAL,
    速动比率 REAL,
    ---现金及现金等价物余额占短期借债的比率 REAL,
    ------流动资产占总负债的比率 REAL,
    现金收入和营业收入的比率 REAL,
    --应收账款周转率 REAL,
    --存货周转率 REAL,
    ------营业利润占营业资金的比率 REAL,
    ------营业收入占营业资金的比率 REAL,
    还原后的净资产收益率（ROCE） REAL,
    --投资收益率 REAL,
    市盈率 REAL,
    本年市值增长率 REAL,
    本年指数增长率 REAL,
    间隔时长 REAL,
    市值增长率 REAL,
    训练标识 INTEGER
);


insert into {0}公司价格分析表
select a.报告时间,
    a.公司代码,
    a.报告类型,
    a.公司简称,
    b.发布时间,
    b.结束时间,
    b.起始周,
    b.报告周,
    b.结束周,
    b.起始周总市值,
    b.报告周总市值,
    b.结束周总市值,
    b.起始周指数,
    b.报告周指数,
    a.营业收入,
    a.归属于上市公司股东的净利润,
    a.归属于上市公司股东的扣除非经常性损益的净利润,
    -------a.经营活动产生的现金流量净额,
    a.员工工资占营业收入比率,
    a.净资产比率,
    a.总资产利润率,
    a.现金分红金额占合并报表中归属于上市公司普通股股东的净利润的比率,
    a.经营活动产生的现金流量净额占净利润的比率,
    ----a.营业收入增长率,
    ----a.归属于上市公司股东的净利润增长率,
    ----a.归属于上市公司股东的扣除非经常性损益的净利润增长率,
    ----a.经营活动产生的现金流量净额增长率,
    -----a.所得税费用占剔除投资收益后利润的比率,
    --a.三费总额占营业收入的比率,
    a.费用总额占营业收入的比率,
    a.资产减值和折旧摊销占营业收入的比率,
    a.营业利润率,
    a.毛利率,
    a.净利率,
    a.平均净资产收益率,
    a.营业收入占平均总资产的比率,
    a.净资产增长率,
    a.研发投入占营业收入的比率,
    a.资本化研发投入的比率,
    a.在建工程占固定资产的比率,
    a.利润总额占生产资本的比率,
    a.商誉占营业收入的比率,
    a.应收账款占营业收入的比率,
    --a.预收总额和应收总额的比率,
    --a.预收款项和应收账款的比率,
    a.流动比率,
    a.速动比率,
    ---a.现金及现金等价物余额占短期借债的比率,
    ------a.流动资产占总负债的比率,
    a.现金收入和营业收入的比率,
    --a.应收账款周转率,
    --a.存货周转率,
    ------a.营业利润占营业资金的比率,
    ------a.营业收入占营业资金的比率,
    a.还原后的净资产收益率（ROCE）,
    --round(a.归属于上市公司股东的净利润/b.报告周总市值,4) as 投资收益率,
    round(b.报告周总市值/a.归属于上市公司股东的净利润,4) as 市盈率,
    b.本年市值增长率,
    b.本年指数增长率,
    b.间隔时长,
    b.市值增长率,
    b.训练标识
from {0}财务分析综合表 a
left join {0}公司价格分析中间表 b
on a.报告时间 = b.报告时间 and a.公司代码 = b.公司代码 and a.报告类型 = b.报告类型
where b.市值增长率 is not NULL and a.报告类型 = '{0}报告';


CREATE INDEX IF NOT EXISTS [{0}公司价格分析表索引] on [{0}公司价格分析表] (
    报告时间,
    公司代码,
    报告类型
);
