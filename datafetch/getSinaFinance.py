#作者：知乎用户
#链接：https://www.zhihu.com/question/29979949/answer/49553763
#来源：知乎
#著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#嗨~我来答题了~虽然题主已经搞定了问题……提问后一周已经搞定了，用了excel power query+yahoo finance api
# 等这周忙完毕业设计回来更新问题…还是非常感谢～！就当练手了~问题的解决办法有很多。利用现有的api挺方便。
# 不过我还是按照题主原来的思路笨办法写写试试。老规矩边做边调边写~#新手 很笨 大神求不喷 新手多交流
# #start coding第一步自然是搜集股票代码…用在线的PDF2DOC网站，然后把13、14、15三类的股票代码复制粘贴到一个文本文档里。
# 像这样…<img data-rawheight="346" data-rawwidth="719"
# src="https://pic3.zhimg.com/50/0a42a5a11a845dd7d1d7387cb6bec6d1_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="719"
# data-original="https://pic3.zhimg.com/0a42a5a11a845dd7d1d7387cb6bec6d1_r.jpg"/>
# 然后我们需要让Python按行读入文本文档里的内容并存入一个列表。很简单。
f=open('stock_num.txt')
stock = []
for line in f.readlines():
    #print(line,end = '')
    line = line.replace('\n','')
    stock.append(line)
f.close()
print(stock)
#<img data-rawheight="250" data-rawwidth="652"
# src="https://pic1.zhimg.com/50/9e55a8ad89aa7b1153d9c83c3bd353ed_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="652"
# data-original="https://pic1.zhimg.com/9e55a8ad89aa7b1153d9c83c3bd353ed_r.jpg"/>
# 然后我们看怎么到网页里找到下载文件的链接。题主想的步骤还是复杂了，分析一下新浪财经的网址构成就好。
# 例如康达尔(000048)年度报告
#http://vip.stock.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/000048/page_type/ndbg.phtml
#网址构成中只要更改stockid/六位数字 就可以进入任一股票的年报页面我们现在已经有了一个名为stock的列表存放了需要下载的股票代码，
# 那我们只要将网址修改一下就好了。
# for each in stock:
#    url='http://vip.stock.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/'+each+'/page_type/ndbg.phtml'
#然后我们需要在年报页面中找到可以下载的PDF地址。Firefox F12。
# <img data-rawheight="52" data-rawwidth="613"
# src="https://pic3.zhimg.com/50/2f371c30be896839800c77658968d9a0_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="613"
# data-original="https://pic3.zhimg.com/2f371c30be896839800c77658968d9a0_r.jpg"/>
# 很显然~地址有了~我们只要在年报列表的页面里匹配所有符合条件的页面就好。但是进入了这个页面并不是最终的PDF下载页面。
# 想要最终获取下载页面还得多走一步。
# <img data-rawheight="160" data-rawwidth="981"
# src="https://pic4.zhimg.com/50/3e95e70daaf2a7d5ebe296d781f6d203_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="981"
# data-original="https://pic4.zhimg.com/3e95e70daaf2a7d5ebe296d781f6d203_r.jpg"/>
# <img data-rawheight="38" data-rawwidth="579"
# src="https://pic2.zhimg.com/50/027004b3dc1e568a330a4b55c8539aa6_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="579"
# data-original="https://pic2.zhimg.com/027004b3dc1e568a330a4b55c8539aa6_r.jpg"/>
# 好，到此为止，我们就一步一步地找到了一个年报的下载地址……那我们需要下载所有股票的所有年报（没有区分近五年）怎么做？
# 两步循环就好。循环股票代码，得到他们的年报列表，再循环年报列表，下载所有年报~我们先来试试下载一支股票的所有年报。
# 还是以康达尔(000048)年度报告为例吧。
import urllib.request
import re
import os

#http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=000048&id=1422814
#http://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESH_STOCK/2009/2009-3/2009-03-10/398698.PDF
url='http://vip.stock.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/'+'000048'+'/page_type/ndbg.phtml'
req = urllib.request.Request(url)
req.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.2; rv:16.0) Gecko/20100101 Firefox/16.0')
page = urllib.request.urlopen(req)
html = page.read().decode('utf-8')
target = r'&id=[_0-9_]{6,7}'
target_list = re.findall(target,html)
print(target_list)
#<img data-rawheight="115" data-rawwidth="652"
# src="https://pic1.zhimg.com/50/750bd06168c8555097d4d57a07cafeb0_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="652"
# data-original="https://pic1.zhimg.com/750bd06168c8555097d4d57a07cafeb0_r.jpg"/>
# 报错也不要害怕~解决就是了。编码错误…去查看网页的编码
# <img data-rawheight="102" data-rawwidth="699"
# src="https://pic3.zhimg.com/50/1fb78e3ff4782fae5e4cc0a79ec44e33_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="699"
# data-original="https://pic3.zhimg.com/1fb78e3ff4782fae5e4cc0a79ec44e33_r.jpg"/>
# 把代码里的utf-8改成gb2312 不重复贴了，重新运行。
# #后面我改成gbk了……编码什么的经常出错，不赘述。自己遇到坑就懂了。
# <img data-rawheight="93"
# data-rawwidth="654"
# src="https://pic1.zhimg.com/50/dd0bb89ada5c568a8e526b0a1b2be829_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="654"
# data-original="https://pic1.zhimg.com/dd0bb89ada5c568a8e526b0a1b2be829_r.jpg"/>
# 得到了所有年报的地址页面。然后我们尝试下载其中一个年报。以康达尔(000048)_公司公告为例。
# 我们新建一个以股票代码命名的文件夹保存PDF。
import urllib.request
import re
import os
os.mkdir('./000048')
target_url='http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=000048'+'&id=712408'
treq = urllib.request.Request(target_url)
treq.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.2; rv:16.0) Gecko/20100101 Firefox/16.0')
tpage = urllib.request.urlopen(treq)
thtml = tpage.read().decode('gbk')
#print(thtml)
file_url = re.search('http://file.finance.sina.com.cn/211.154.219.97:9494/.*?PDF',thtml)
print(file_url.group(0))
local = './'+'000048'+'/'+file_url.group(0).split("/")[-1]+'.pdf'
urllib.request.urlretrieve(file_url.group(0),local,None)
#结果：<img data-rawheight="90" data-rawwidth="280"
# src="https://pic4.zhimg.com/50/c7348e4436b5efb5cf16260d6546049c_hd.jpg"
# class="content_image" width="280"/><img data-rawheight="463" data-rawwidth="990"
# src="https://pic4.zhimg.com/50/e83b469474a4be52f5443611307ade18_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="990"
# data-original="https://pic4.zhimg.com/e83b469474a4be52f5443611307ade18_r.jpg"/>
# ————————————至此我们可以确定能下载成功————————————开始完善循环。下载某一股票所有年报。
import urllib.request
import re
import os

url='http://vip.stock.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/'+'000048'+'/page_type/ndbg.phtml'
req = urllib.request.Request(url)
req.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.2; rv:16.0) Gecko/20100101 Firefox/16.0')
page = urllib.request.urlopen(req)
html = page.read().decode('gbk')
target = r'&id=[_0-9_]{6}'
target_list = re.findall(target,html)
os.mkdir('./000048')
for each in target_list:
    print(each)
    target_url='http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=600616'+each
    treq = urllib.request.Request(target_url)
    treq.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.2; rv:16.0) Gecko/20100101 Firefox/16.0')
    tpage = urllib.request.urlopen(treq)
    thtml = tpage.read().decode('gbk')
    #print(thtml)
    file_url = re.search('http://file.finance.sina.com.cn/211.154.219.97:9494/.*?PDF',thtml)
    print(file_url.group(0))
    local = './000048/'+file_url.group(0).split("/")[-1]+'.pdf'
    #写入一个空文件站位，实际使用时使用urlretrieve可以下载文件
    open(local, 'wb').write(b'success')
    #urllib.request.urlretrieve(file_url.group(0),local,None)
#运行…<img data-rawheight="221" data-rawwidth="672"
# src="https://pic1.zhimg.com/50/813c4dfd5c1fdf093328a8bd6b1cd025_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="672"
# data-original="https://pic1.zhimg.com/813c4dfd5c1fdf093328a8bd6b1cd025_r.jpg"/>
# 果然没有一帆风顺的事情。不过不要怕，我们看一下是哪里出错了。
# 打开&id=1394915这个页面。康达尔(000048)_公司公告
# <img data-rawheight="192" data-rawwidth="1010"
# src="https://pic4.zhimg.com/50/892973f3b31eb356c6ada590e8d8d4fa_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="1010"
# data-original="https://pic4.zhimg.com/892973f3b31eb356c6ada590e8d8d4fa_r.jpg"/>
# 诶，这个页面里根本没有下载链接。也就是说我们在匹配下载地址的时候返回了None。
# 我们使用try语句处理异常，遇到没有下载链接的页面输出“失效”。修改代码如下：
import urllib.request
import re
import os

url='http://vip.stock.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/'+'000048'+'/page_type/ndbg.phtml'
req = urllib.request.Request(url)
req.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.2; rv:16.0) Gecko/20100101 Firefox/16.0')
page = urllib.request.urlopen(req)
html = page.read().decode('gbk')
target = r'&id=[_0-9_]{6,7}'
target_list = re.findall(target,html)
os.mkdir('./000048')
for each in target_list:
    print(each)
    target_url='http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=600616'+each
    treq = urllib.request.Request(target_url)
    treq.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.2; rv:16.0) Gecko/20100101 Firefox/16.0')
    tpage = urllib.request.urlopen(treq)
    thtml = tpage.read().decode('gbk')
    #print(thtml)
    try:
        file_url = re.search('http://file.finance.sina.com.cn/211.154.219.97:9494/.*?PDF',thtml)
        print(file_url.group(0))
        local = './000048/'+file_url.group(0).split("/")[-1]+'.pdf'
        #写入一个空文件站位，实际使用时使用urlretrieve可以下载文件
        open(local, 'wb').write(b'success')
        #urllib.request.urlretrieve(file_url.group(0),local,None)
    except:
        print('失效')
#<img data-rawheight="515" data-rawwidth="651"
# src="https://pic1.zhimg.com/50/05b848400f253a233da750ae81da19d6_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="651"
# data-original="https://pic1.zhimg.com/05b848400f253a233da750ae81da19d6_r.jpg"/>
# <img data-rawheight="262" data-rawwidth="266"
# src="https://pic2.zhimg.com/50/8e4cd7ae20dc5981e2925aaf597c8056_hd.jpg"
# class="content_image" width="266"/>看上去很完美……加上外层stockid再循环试试…
import urllib.request
import re
import os

f=open('stock_num.txt')
stock = []
for line in f.readlines():
    #print(line,end = '')
    line = line.replace('\n','')
    stock.append(line)
f.close()
#print(stock)

for each in stock:
    url='http://vip.stock.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/'+each+'/page_type/ndbg.phtml'
    req = urllib.request.Request(url)
    req.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.2; rv:16.0) Gecko/20100101 Firefox/16.0')
    page = urllib.request.urlopen(req)

    html = page.read().decode('gbk')
    target = r'&id=[_0-9_]{6}'
    target_list = re.findall(target,html)
    os.mkdir('./'+each)
    sid = each
    #print(target_list)
    for each in target_list:
        #print(a)
        #print(each)
        target_url='http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid='+sid+each
        #print(target_url)
        treq = urllib.request.Request(target_url)
        treq.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.2; rv:16.0) Gecko/20100101 Firefox/16.0')
        tpage = urllib.request.urlopen(treq)
        thtml = tpage.read().decode('gbk')
        #print(thtml)
        file_url = re.search('http://file.finance.sina.com.cn/211.154.219.97:9494/.*?PDF',thtml)
        try:
            #print(file_url.group(0))
            local = './'+sid+'/'+file_url.group(0).split("/")[-1]+'.pdf'
            #调试用作文件占位
            open(local, 'wb').write(b'success')
            #print(local)
            #urllib.request.urlretrieve(file_url.group(0),local,None)
        except:
            print('PDF失效;'+target_url)
#为了控制台输出的整洁，PDF下载成功的话就不在控制台显示了，只输出地址失效的信息。
# <img data-rawheight="437" data-rawwidth="653"
# src="https://pic2.zhimg.com/50/2dadf2d3587ece4e74ab33341fdadbc5_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="653"
# data-original="https://pic2.zhimg.com/2dadf2d3587ece4e74ab33341fdadbc5_r.jpg"/>
# ………………又是编码的错误。遇到这种情况也是没办法了。同样的网页别人都能就它不能，还是用try先跳过，最后统一处理吧。
# 为了以防万一我直接把所有涉及页面操作的地方全加了try。最终代码如下：
import urllib.request
import re
import os

f=open('stock_num.txt')
stock = []
for line in f.readlines():
    #print(line,end = '')
    line = line.replace('\n','')
    stock.append(line)
#print(stock)
f.close()
#print(stock)

for each in stock:
    url='http://vip.stock.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/'+each+'/page_type/ndbg.phtml'
    req = urllib.request.Request(url)
    req.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.2; rv:16.0) Gecko/20100101 Firefox/16.0')
    page = urllib.request.urlopen(req)
    try:
        html = page.read().decode('gbk')
        target = r'&id=[_0-9_]{6}'
        target_list = re.findall(target,html)
        os.mkdir('./'+each)
        sid = each
        #print(target_list)
        for each in target_list:
            #print(a)
            #print(each)
            target_url='http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid='+sid+each
            #print(target_url)
            treq = urllib.request.Request(target_url)
            treq.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.2; rv:16.0) Gecko/20100101 Firefox/16.0')
            tpage = urllib.request.urlopen(treq)
            try:
                thtml = tpage.read().decode('gbk')
                #print(thtml)
                file_url = re.search('http://file.finance.sina.com.cn/211.154.219.97:9494/.*?PDF',thtml)
                try:
                    #print(file_url.group(0))
                    local = './'+sid+'/'+file_url.group(0).split("/")[-1]+'.pdf'
                    #调试用作文件占位
                    #open(local, 'wb').write(b'success')
                    #print(local)
                    urllib.request.urlretrieve(file_url.group(0),local,None)
                except:
                    print('PDF失效;'+target_url)
            except:
                print('年报下载页面编码错误;'+target_url)
    except:
        print('年报列表页面编码错误;'+url)
#到这里就已经解决了问题~自己试了一下应该是成功的。校园网流量不多了就没有下载全部的PDF而是拿空文件占位跑了一下结果。
# <img data-rawheight="511"
# data-rawwidth="649"
# src="https://pic3.zhimg.com/50/354caffa9d29b695e4943236d74e4613_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="649"
# data-original="https://pic3.zhimg.com/354caffa9d29b695e4943236d74e4613_r.jpg"/>
# <img data-rawheight="544" data-rawwidth="482"
# src="https://pic3.zhimg.com/50/b69c3cd0d05963051bbb0df11d35fb76_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="482"
# data-original="https://pic3.zhimg.com/b69c3cd0d05963051bbb0df11d35fb76_r.jpg"/>
# <img data-rawheight="264" data-rawwidth="415"
# src="https://pic3.zhimg.com/50/bd4592c862278e79e864fa1b726cddc5_hd.jpg"
# class="content_image" width="415"/>
# 最后我手动把控制台的报错信息输入到excel里分列。
# <img data-rawheight="472" data-rawwidth="843"
# src="https://pic4.zhimg.com/50/06777a09c33da35d8670794c1edb4b3b_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="843"
# data-original="https://pic4.zhimg.com/06777a09c33da35d8670794c1edb4b3b_r.jpg"/>
# PDF失效的情况可以忽视。年报下载页面编码错误的情况不太多，我测试的时候遇到了13处。
# 可以直接手动解决。PS。遇到Errno 11001的报错应该是爬虫被拒绝了。等一会儿重新打开就好。
# 一次性的解决办法应该是完善headers和加代理。这次的工作量不大，遇到无法访问的情况也不多，就没有写。
# <img data-rawheight="63" data-rawwidth="591"
# src="https://pic1.zhimg.com/50/c54485902f16bea9e04178eeda3ef35a_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="591"
# data-original="https://pic1.zhimg.com/c54485902f16bea9e04178eeda3ef35a_r.jpg"/>
# To-dos：还是流水式的写法……自己还是不大会模块化的实现要学着用requests代替urllib.request，
# 学着使用BeautifulSoup报错信息或许可以自动写入excel……
# 然后程序自动处理报错的那些页面……编码问题最烦人了自己没怎么遇到网络问题大概是因为数据量小…
# 除了刚开始测试下载了整个PDF后面都是在拿空文件占位，数据量大容易被服务器封掉，
# 还是要完善headers和proxyList
# 编辑于 2015-06-01​赞同 283​​34 条评论​分享​收藏​喜欢收起​更多回答八爪鱼采集器​已认证的官方帐号25 人赞同了该回答关于这个需求，
# python略微麻烦了一点，用八爪鱼就十分简单了，嗖嗖的就能把题主需要的数据下载下来。下面给大家介绍具体如何操作：1、
# 这里感谢段小草大佬提供的搜集股票代码思路，用在线的PDF2DOC网站，然后把13、14、15三类的股票代码复制粘贴到一个文本文档里。
# <img src="https://pic1.zhimg.com/50/v2-869c69d080d0bcd5afbb328bbd78e285_hd.jpg"
# data-caption="" data-size="normal" data-rawwidth="1355" data-rawheight="697"
# class="origin_image zh-lightbox-thumb" width="1355"
# data-original="https://pic1.zhimg.com/v2-869c69d080d0bcd5afbb328bbd78e285_r.jpg"/>
# 2、分析年报的下载链接组成，通过（*ST康达(000048)股票股价,行情,新闻,财报数据_新浪财经_新浪网）
# 这个网页可以找到下载年报的网页地址为：
# http://vip.stock.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/000048/page_type/ndbg.phtml
# 注释：观察发现/stockid/后面的/000048/就是第一步转化而来的股票地址，替换一下就能访问不同骨片的年报下载网页。
# 类似的还可以去下载（如上诉步骤将股票代码替换相应位置就行）：
# 股票公告：http://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllBulletin/stockid/000048.phtml
# 半年报：http://vip.stock.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/000048/page_type/zqbg.phtml
# 一季报：http://vip.stock.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/000048/page_type/yjdbg.phtml
# 三季报：http://vip.stock.finance.sina.com.cn/corp/go.php/vCB_Bulletin/stockid/000048/page_type/sjdbg.phtml3、
# 使用八爪鱼自定义模式将股票年报下载链接采集下来，将年报下载页面链接与股票代码拼接可以使用八爪鱼网址批量生成功能。
# <img src="https://pic4.zhimg.com/50/v2-4858a6356b98d5f76d7cc28f0da0b09b_hd.gif"
# data-size="normal" data-rawwidth="1282" data-rawheight="794"
# data-thumbnail="https://pic4.zhimg.com/50/v2-4858a6356b98d5f76d7cc28f0da0b09b_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="1282"
# data-original="https://pic4.zhimg.com/v2-4858a6356b98d5f76d7cc28f0da0b09b_r.jpg"/>
# 年报下载页面链接与股票代码拼接后面只需做一个循环点击+提取数据就能批量将地址采集下载，给大家看下采集效果
# <img src="https://pic3.zhimg.com/50/v2-0ed79c56f45feafcca32c969a7b1efa3_hd.gif"
# data-caption="" data-size="normal" data-rawwidth="1282" data-rawheight="794"
# data-thumbnail="https://pic3.zhimg.com/50/v2-0ed79c56f45feafcca32c969a7b1efa3_hd.jpg"
# class="origin_image zh-lightbox-thumb" width="1282"
# data-original="https://pic3.zhimg.com/v2-0ed79c56f45feafcca32c969a7b1efa3_r.jpg"/>
# 4、有了下载地址，接下来就很好办了，使用下载工具批量进行下载，这里推荐八爪鱼批量下载工具，最终效果如下。​
# <img src="https://pic1.zhimg.com/50/v2-e29f6c0f75ca6bb8c3f0037ff9a4ddb7_hd.jpg" data-size="normal"
# data-rawwidth="1200" data-rawheight="799" class="origin_image zh-lightbox-thumb" width="1200"
# data-original="https://pic1.zhimg.com/v2-e29f6c0f75ca6bb8c3f0037ff9a4ddb7_r.jpg"/>
# 八爪鱼批量下载工具满满都是自己需要的年报，有点小激动
# <img src="https://pic3.zhimg.com/50/v2-11e889926e38c1d45768524ed978a61d_hd.jpg"
# data-caption="" data-size="normal" data-rawwidth="1587"
# data-rawheight="932" class="origin_image zh-lightbox-thumb" width="1587"
# data-original="https://pic3.zhimg.com/v2-11e889926e38c1d45768524ed978a61d_r.jpg"/>
# 欢迎大家进行点赞，点赞超过50，我把文中的：
# 1、八爪鱼批量采集新浪财经网指定企业年报的爬虫规则2、按照题主需求下载好的企业年报分享出来，发给大家最后附上软件的下载地址：