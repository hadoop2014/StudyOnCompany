drop table if exists 年度公司价值分析表;
create table if not exists 年度公司价值分析表
as
select a.*,
    b.市盈率,
    b.本年市值增长率,
    b.本年指数增长率,
    b.间隔时长,
    b.市值增长率,
    b.预测市值增长率
from 年度财务分析综合表 a
left join 年度公司价格预测表 b
on a.报告时间 = b.报告时间 and a.公司代码 = b.公司代码 and a.报告类型 = b.报告类型;


UPDATE 年度公司价值分析表
SET 公司投资等级 =
(
    select round(a.预测市值增长率,2) - round(a.市值增长率,2) as 预测市值增长率
    from
    (
        select x.公司代码,
            x.报告类型,
            x.报告时间,
            x.预测市值增长率,
            x.市值增长率
        from 年度公司价格预测表 x
        left join
        (
            select 公司代码,报告类型,
                max(报告时间) as 报告时间
            from 年度公司价格预测表
            group by 公司代码,报告类型
        )y
        where x.公司代码 = y.公司代码 and x.报告类型 = y.报告类型 and x.报告时间 = y.报告时间
    )a
    where a.公司代码 = 年度公司价值分析表.公司代码 and a.报告类型 = 年度公司价值分析表.报告类型
);


CREATE INDEX IF NOT EXISTS [年度公司价值分析表索引] on [年度公司价值分析表] (
    报告时间,
    公司代码,
    报告类型
);


