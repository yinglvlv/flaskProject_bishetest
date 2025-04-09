import requests
import os
import time
import random
from bs4 import BeautifulSoup

# 唯一国家列表（22个，USA和United States合并）
f1_countries_urls = {
    "Australia": "australia",
    "Austria": "austria",
    "Azerbaijan": "azerbaijan",
    "Bahrain": "bahrain",
    "Belgium": "belgium",
    "Brazil": "brazil",
    "Canada": "canada",
    "China": "the-people-s-republic-of-china",
    "France": "france",
    "Hungary": "hungary",
    "Italy": "italy",
    "Japan": "japan",
    "Mexico": "mexico",
    "Monaco": "monaco",
    "Netherlands": "the-netherlands",
    "Portugal": "portugal",
    "Qatar": "qatar",
    "Russia": "russia",
    "Saudi_Arabia": "saudi-arabia",
    "Singapore": "singapore",
    "Spain": "spain",
    "Turkey": "turkey",
    "UAE": "the-united-arab-emirates",
    "UK": "the-united-kingdom",
    "USA": "the-united-states"  # USA和United States都映射到united-states
}

# 创建保存国旗的文件夹
save_folder = "F1_Flags_2021-2025"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 设置请求头
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"
}


# 下载函数（带重试机制）
def download_flag(country, url, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            flag_img = soup.find("img", src=lambda x: x and "/data/flags/" in x)
            if flag_img and "src" in flag_img.attrs:
                img_url = "https://flagpedia.net" + flag_img["src"]
                print(f"找到图片URL: {img_url}")

                img_data = requests.get(img_url, headers=headers, timeout=10).content
                file_name = os.path.join(save_folder, f"{country}.png")

                with open(file_name, "wb") as f:
                    f.write(img_data)
                print(f"已下载: {country} 的国旗 -> {file_name}")
                return True
            else:
                print(f"未找到 {country} 的国旗图片，检查网页结构或URL: {url}")
                imgs = soup.find_all("img")
                for img in imgs:
                    print(f"图像元素: {img}")
                return False

        except requests.RequestException as e:
            print(f"下载 {country} 失败 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(random.uniform(2, 5))
            else:
                print(f"放弃 {country} 的下载，已达最大重试次数")
                return False


# 主循环
for country, url_path in f1_countries_urls.items():
    url = f"https://flagpedia.net/{url_path}"
    print(f"正在访问: {url}")
    download_flag(country, url)
    time.sleep(random.uniform(1, 3))

print("所有国旗爬取完成！图片保存在: ", save_folder)