drop table if exists 年度公司价格分析表;
create table if not exists 年度公司价格分析表 (
    报告时间 DATE NOT NULL,
    公司代码 INTEGER NOT NULL,
    报告类型 CHAR(20) NOT NULL,
    公司简称 CHAR(10),
    发布时间 DATE,
    文件名 CHAR(20),
    报告周 CHAR(10),
    报告周总市值 REAL,
    结束周 CHAR(10),
    结束周总市值 REAL,
    增长率 REAL
);

insert into 年度公司价格分析表
select
    a.报告时间,
    a.公司代码,
    a.报告类型,
    a.公司简称,
    a.发布时间,
    a.文件名,
    a.报告周,
    b.周平均总市值 as 报告周总市值,
    a.结束周,
    c.周平均总市值 as 结束周总市值,
    round((c.周平均总市值 - b.周平均总市值)/b.周平均总市值, 4) as 增长率
from
(
    select x.报告时间,
        x.公司代码,
        x.公司简称,
        x.报告类型,
        x.发布时间,
        x.文件名,
        strftime('%Y-%W', x.发布时间) as 报告周,
        case when strftime('%Y-%W',x.发布时间, '+1 year') > z.截止周
            then z.截止周
            else strftime('%Y-%W',x.发布时间, '+1 year') end
            as 结束周,
        strftime('%Y-%W',x.发布时间, '+1 year') as 最大周
        -- row_number() OVER(
        --    PARTITION BY x.公司代码, x.报告时间 ORDER BY x.发布时间  -- RANGER BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
        -- ) AS 去重标识
    from
    (
        select bb.*
        from
        (
            -- 同一个公司同一年度可能会发布两次财报, 第二次为第一次发布财报的修正, 但是我们认为第一次发布的财报影响更大, 因此取第一次发布的时间为准
            select 公司代码, 报告时间, 报告类型, min(发布时间) as 发布时间
            from 财报发布信息
            group by 公司代码, 报告时间, 报告类型
        )aa
        left join 财报发布信息 bb
        on aa.公司代码 = bb.公司代码 and aa.报告时间 = bb.报告时间 and aa.报告类型 = bb.报告类型 and aa.发布时间 = bb.发布时间
    )x
    left join
    (
        select 公司代码,
            max(报告周) as 截止周
        from
        (
            select 公司代码,
                strftime('%Y-%W', 报告时间) as 报告周
            from 股票交易数据
        )y
        group by 公司代码
    )z
    where x.公司代码 = z.公司代码 and x.报告类型 = '年度报告'
)a
left join
(
    select x.报告周,
        x.公司代码,
        x.公司简称,
        round(avg(x.总市值), 0) as 周平均总市值
    from
    (
        select *,
            strftime('%Y-%W', 报告时间) as 报告周
        from 股票交易数据
    )x
    group by 报告周, 公司代码, 公司简称
)b
left join
(
    select x.报告周,
        x.公司代码,
        x.公司简称,
        round(avg(x.总市值), 0) as 周平均总市值
    from
    (
        select *,
            strftime('%Y-%W', 报告时间) as 报告周
        from 股票交易数据
    )x
    group by 报告周, 公司代码, 公司简称
)c
where (a.公司代码 = b.公司代码 and a.报告周 = b.报告周)
    and (a.公司代码 = c.公司代码 and a.结束周 = c.报告周);


CREATE INDEX IF NOT EXISTS [年度公司价格分析表索引] on [年度公司价格分析表] (
    报告时间,
    公司代码,
    报告类型
);


