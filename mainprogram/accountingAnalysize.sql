

select a.报告时间,a.股票代码,a.报告类型,a.股票简称,a.公司名称,a.营业收入,a.归属于上市公司股东的净利润,b.在职员工的数量合计,replace(a.营业收入,',',''),
       round(replace(a.营业收入,',','')/replace(b.在职员工的数量合计,',',''),0) as 人均营业收入
from 主要会计数据 as a, 关键数据表 as b
where a.报告时间 = b.报告时间 and a.股票代码 = b.股票代码 and a.报告类型 = b.报告类型 and a.股票简称 = b.股票简称;