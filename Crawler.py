import requests
import re
import os
from lxml import etree

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}

#获取网站图片地址函数模块
def get_page(num):
    img_list = []
    for i in range((num // 35) + 1):
        url = f'https://cn.bing.com/images/search?q=%E7%94%B5%E9%92%BB&qs=n&form=QBIR&sp=-1&lq=0&pq=%E7%94%B5%E9%92%BB&sc=10-2&cvid=77D75FC760E5411EB95181E9BF8837AF&first=1'
        r = requests.get(url, headers=headers)  # 刚才复制的链接地址
        html = r.text
        html = etree.HTML(html)
        conda_list = html.xpath('//a[@class="iusc"]/@m')
        for j in conda_list:
            pattern = re.compile(r'"murl":"(.*?)"')
            img_url = re.findall(pattern, j)[0]
            img_list.append(img_url)
    return img_list

#下载网站图片
def download(path, img_list):
    for i in range(len(img_list)):
        img_url = img_list[i]
        print(f'正在爬取: {img_url}')
        img_floder = 'D:/resource/Project/Competition/人工智能/dataset/电钻/' + keyword  # 爬取图片存放的位置并命名
        if not os.path.exists(img_floder):
            os.makedirs(img_floder)
        try:
            with open(f'{img_floder}/{i}.jpg', 'wb') as f:
                img_content = requests.get(img_url).content
                f.write(img_content)
        except:
            continue

#主函数
if __name__ == '__main__':
    num = 20
    keyword = ''  # 文件名
    path = 'D:/resource/Project/Competition/人工智能/dataset/电钻'  # 存放位置
    img_list = get_page(num)
    download(path, img_list)