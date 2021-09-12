# -*- coding: utf-8 -*

from bs4 import BeautifulSoup
import requests
import sys
import time
def crawl():
    url = "https://www.ptt.cc/bbs/Beauty/index.html"
    lastdate_of_2019 = False
    while not lastdate_of_2019:
        payload = {
            'from': url.replace("https://www.ptt.cc", ""),
            'yes': 'yes'
        }
        rs = requests.session()
        res = rs.post("https://www.ptt.cc/ask/over18", data=payload)
        res = rs.get(url)
        content = res.text
        soup = BeautifulSoup(content, 'html.parser')
        date = soup.find_all(class_="date")
        for d in date:
            if d.get_text().find("12/01") > -1:
                lastdate_of_2019 = True
        if lastdate_of_2019 == False:
            btns = soup.find_all(class_="btn wide")
            for b in btns:
                if b.get_text().find("上頁") > -1:
                    url = "https://www.ptt.cc" + b.get('href')
        time.sleep(0.05)

    lastdate_of_2018 = False
    while not lastdate_of_2018:
        payload = {
            'from': url.replace("https://www.ptt.cc", ""),
            'yes': 'yes'
        }
        rs = requests.session()
        res = rs.post("https://www.ptt.cc/ask/over18", data=payload)
        res = rs.get(url)
        content = res.text
        soup = BeautifulSoup(content, 'html.parser')
        date = soup.find_all(class_="date")
        for d in date:
            if d.get_text().find("12/31") > -1:
                lastdate_of_2018 = True
        if lastdate_of_2018 == False:
            btns = soup.find_all(class_="btn wide")
            for b in btns:
                if b.get_text().find("上頁") > -1:
                    url = "https://www.ptt.cc" + b.get('href')
        time.sleep(0.05)

    dic = {}
    dic_pop = {}
    #find 12/31
    payload = {
        'from': url.replace("https://www.ptt.cc", ""),
        'yes': 'yes'
    }
    rs = requests.session()
    res = rs.post("https://www.ptt.cc/ask/over18", data=payload)
    res = rs.get(url)
    content = res.text
    soup = BeautifulSoup(content, 'html.parser')
    rent = soup.find_all(class_="r-ent")
    for r in rent:
        d = r.find(class_="date").get_text()
        if d.find("1/01") > -1:
            title = r.find(class_="title").get_text().split("\n")[1]
            u = "https://www.ptt.cc" + r.find(class_="title").find('a').get('href')
            day = d.split("/")[0].split(" ")[-1]+d.split("/")[1]
            if day not in list(dic.keys()):
                dic[day] = []
            dic[day].append([title,u])
            if r.find(class_="nrec").get_text().find('爆') > -1:
                if day not in list(dic_pop.keys()):
                    dic_pop[day] = []
                dic_pop[day].append([title, u])
    btns = soup.find_all(class_="btn wide")
    for b in btns:
        if b.get_text().find("下頁") > -1:
            url = "https://www.ptt.cc" + b.get('href')

    lastdate_of_2019 = False
    while not lastdate_of_2019:
        payload = {
            'from': url.replace("https://www.ptt.cc", ""),
            'yes': 'yes'
        }
        rs = requests.session()
        res = rs.post("https://www.ptt.cc/ask/over18", data=payload)
        res = rs.get(url)
        content = res.text
        soup = BeautifulSoup(content, 'html.parser')
        rent = soup.find_all(class_="r-ent")
        for r in rent:
            d = r.find(class_="date").get_text()
            if d.find("12/31") == -1:
                title = r.find(class_="title").get_text().split("\n")[1]
                try:
                    u = "https://www.ptt.cc" + r.find(class_="title").find('a').get('href')
                    day = d.split("/")[0].split(" ")[-1] + d.split("/")[1]
                    if day not in list(dic.keys()):
                        dic[day] = []
                    dic[day].append([title, u])
                    if r.find(class_="nrec").get_text().find('爆') > -1:
                        if day not in list(dic_pop.keys()):
                            dic_pop[day] = []
                        dic_pop[day].append([title, u])
                except:
                    print(r)
            else:
                lastdate_of_2019 = True
        if lastdate_of_2019 == False:
            btns = soup.find_all(class_="btn wide")
            for b in btns:
                if b.get_text().find("下頁") > -1:
                    url = "https://www.ptt.cc" + b.get('href')
        time.sleep(0.05)
    #find 1/1
    firstdate_of_2019 = False
    while not firstdate_of_2019:
        payload = {
            'from': url.replace("https://www.ptt.cc", ""),
            'yes': 'yes'
        }
        rs = requests.session()
        res = rs.post("https://www.ptt.cc/ask/over18", data=payload)
        res = rs.get(url)
        content = res.text
        soup = BeautifulSoup(content, 'html.parser')
        rent = soup.find_all(class_="r-ent")
        for r in rent:
            d = r.find(class_="date").get_text()
            if d.find("1/01") == -1:
                if d.find("12/30") == -1:
                    title = r.find(class_="title").get_text().split("\n")[1]
                    try:
                        u = "https://www.ptt.cc" + r.find(class_="title").find('a').get('href')
                        day = d.split("/")[0].split(" ")[-1] + d.split("/")[1]
                        if day not in list(dic.keys()):
                            dic[day] = []
                        dic[day].append([title, u])
                        if r.find(class_="nrec").get_text().find('爆') > -1:
                            if day not in list(dic_pop.keys()):
                                dic_pop[day] = []
                            dic_pop[day].append([title, u])
                    except:
                        print(r)
            else:
                firstdate_of_2019 = True
        if firstdate_of_2019 == False:
            btns = soup.find_all(class_="btn wide")
            for b in btns:
                if b.get_text().find("下頁") > -1:
                    url = "https://www.ptt.cc" + b.get('href')
        time.sleep(0.05)

    #print(len(dic))

    with open('all_articles.txt', 'w', encoding='UTF-8') as out_file, open('all_popular.txt', 'w', encoding='UTF-8') as outpop_file:
        k_lst = dic.keys()
        key_list = list(k_lst)[:]
        num = 0
        for i in key_list:
            for j in range(len(dic[i])):
                if dic[i][j][0].find("[公告]") == -1:
                    if num == 0:
                        out_file.write(i+","+dic[i][j][0]+","+dic[i][j][1])
                        num = 1
                    else:
                        out_file.write("\n" + i + "," + dic[i][j][0] + "," + dic[i][j][1])


        kpop_lst = dic_pop.keys()
        keypop_list = list(kpop_lst)[:]
        num = 0
        for i in keypop_list:
            for j in range(len(dic_pop[i])):
                if dic_pop[i][j][0].find("[公告]") == -1:
                    if num == 0:
                        outpop_file.write(i + "," + dic_pop[i][j][0] + "," + dic_pop[i][j][1])
                        num = 1
                    else:
                        outpop_file.write("\n" + i + "," + dic_pop[i][j][0] + "," + dic_pop[i][j][1])


def push_st_ed(start_date,end_date):
    sd = int(start_date)
    ed = int(end_date)
    like_dic = {}
    boo_dic = {}
    like = 0
    boo = 0


    with open('all_articles.txt', 'r', encoding='UTF-8') as in_file:
        all_lines = in_file.readlines()
    for line in all_lines:
        line = line.split(',')
        if int(line[0]) < sd:
            continue
        elif int(line[0]) > ed:
            break

        url = line[-1].replace("\n","")
        payload = {
            'from': url.replace("https://www.ptt.cc", ""),
            'yes': 'yes'
        }
        rs = requests.session()
        res = rs.post("https://www.ptt.cc/ask/over18", data=payload)
        res = rs.get(url)
        content = res.text
        soup = BeautifulSoup(content, 'html.parser')
        contain = soup.find_all(id="main-container")
        for c in contain:
            # print(c.get_text())
            all_text = c.get_text()
            if all_text.find('※ 發信站') == -1:
                #print(url)
                continue
            push_user = soup.find_all(class_="push")
            for p in push_user:
                #pushtag = p.find(class_="push-tag").get_text()
                pushtag = p.find(class_="push-tag")
                if pushtag is not None:
                    pushtag = pushtag.get_text()

                    userid = p.find(class_="f3 hl push-userid").get_text()
                    if pushtag.find("推") > -1:
                        if userid not in list(like_dic.keys()):
                            like_dic[userid] = 0
                        like_dic[userid] = like_dic[userid] + 1
                        like = like + 1

                    if pushtag.find("噓") > -1:
                        if userid not in list(boo_dic.keys()):
                            boo_dic[userid] = 0
                        boo_dic[userid] = boo_dic[userid] + 1
                        boo = boo + 1
                '''else:
                    print(url)'''
        time.sleep(0.05)

    sort_like = sorted(like_dic.items(), key=lambda d: d[0])
    sort_like = sorted(sort_like, key=lambda d: d[1], reverse=True)
    sort_boo = sorted(boo_dic.items(), key=lambda d: d[0])
    sort_boo = sorted(sort_boo, key=lambda d: d[1], reverse=True)

    with open('push[' + str(sd) + '-' + str(ed) + '].txt', 'w') as out_file:
        out_file.write('all like: ' + str(like) + '\n')
        out_file.write('all boo: ' + str(boo) + '\n')
        for i in range(0, 10):
            if i == 0:
                out_file.write('like #' + str(i + 1) + ': ' + sort_like[i][0] + ' ' + str(sort_like[i][1]))
            else:
                out_file.write('\nlike #' + str(i + 1) + ': ' + sort_like[i][0] + ' ' + str(sort_like[i][1]))
        for i in range(0, 10):
            out_file.write('\nboo #' + str(i + 1) + ': ' + sort_boo[i][0] + ' ' + str(sort_boo[i][1]))

def popular(start_date,end_date):
    sd = int(start_date)
    ed = int(end_date)
    pop_num = 0
    url_lst = []
    with open('all_popular.txt', 'r', encoding='UTF-8') as in_file:
        all_lines = in_file.readlines()
    for line in all_lines:
        line = line.split(',')
        if int(line[0]) < sd:
            continue
        elif int(line[0]) > ed:
            break

        pop_num = pop_num + 1

        url = line[-1].replace("\n","")
        payload = {
            'from': url.replace("https://www.ptt.cc", ""),
            'yes': 'yes'
        }
        rs = requests.session()
        res = rs.post("https://www.ptt.cc/ask/over18", data=payload)
        #print(url)
        res = rs.get(url)
        content = res.text
        # print(content)
        soup = BeautifulSoup(content, 'html.parser')
        contain = soup.find_all(id="main-container")
        for c in contain:
            # print(c.get_text())
            all_text = c.get_text()
            if all_text.find('※ 發信站') == -1:
                #print(url)
                continue
            img = soup.find_all('a')
            #print(push_user)
            #print(img)
            for i in img:
                u = i.get('href')
                if u.endswith(".jpg") or u.endswith(".JPG") or \
                   u.endswith(".jpeg") or u.endswith(".JPEG") or \
                   u.endswith(".png") or u.endswith(".PNG") or \
                   u.endswith(".gif") or u.endswith(".GIF"):
                    url_lst.append(u)
        time.sleep(0.05)

    with open('popular[' + str(sd) + '-' + str(ed) + '].txt', 'w') as out_file:
        out_file.write('number of popular articles: ' + str(pop_num) + '\n')
        num = 0
        for l in url_lst:
            if num == 0:
                out_file.write(l)
                num = 1
            else:
                out_file.write('\n' + l)

def keyword(key,start_date,end_date):
    sd = int(start_date)
    ed = int(end_date)
    keyw = key.split("{")[-1].split("}")[0]
    url_lst = []

    with open('all_articles.txt', 'r', encoding='UTF-8') as in_file:
        all_lines = in_file.readlines()
    for line in all_lines:
        line = line.split(',')
        if int(line[0]) < sd:
            continue
        elif int(line[0]) > ed:
            break

        url = line[-1].replace("\n","")
        payload = {
            'from': url.replace("https://www.ptt.cc", ""),
            'yes': 'yes'
        }
        rs = requests.session()
        res = rs.post("https://www.ptt.cc/ask/over18", data=payload)
        res = rs.get(url)
        content = res.text
        soup = BeautifulSoup(content, 'html.parser')
        contain = soup.find_all(id="main-container")
        for c in contain:
            #print(c.get_text())
            all_text = c.get_text()
            if all_text.find('※ 發信站') == -1:
                #print("no 發信站")
                #print(url)
                continue

            text1 = all_text.split("※ 發信站:")[0].split("本網站已依台灣網站內容分級規定處理。此區域為限制級，未滿十八歲者不得瀏覽。")[-1]
            if text1.find("--") == -1:
                #print("no --")
                #print(url)
                text = "".join(text1)
            else:
                text = "".join(text1.split("--")[:-1])
            #print(text)
            if text.find(keyw) > -1:#台北玉山
                #print(url)
                img = soup.find_all('a')
                for i in img:
                    u = i.get('href')
                    if u.endswith(".jpg") or u.endswith(".JPG") or \
                            u.endswith(".jpeg") or u.endswith(".JPEG") or \
                            u.endswith(".png") or u.endswith(".PNG") or \
                            u.endswith(".gif") or u.endswith(".GIF"):
                        url_lst.append(u)
            time.sleep(0.05)

    with open('keyword('+keyw+')[' + str(sd) + '-' + str(ed) + '].txt', 'w') as out_file:
        num = 0
        for l in url_lst:
            if num == 0:
                out_file.write(l)
                num = 1
            else:
                out_file.write('\n' + l)




if sys.argv[1] == 'crawl':
    crawl()
elif sys.argv[1] == 'push':
    push_st_ed(sys.argv[2],sys.argv[3])
elif sys.argv[1] == 'popular':
    popular(sys.argv[2],sys.argv[3])
elif sys.argv[1] == 'keyword':
    keyword(sys.argv[2],sys.argv[3],sys.argv[4])

