# -*- coding: utf-8 -*-
# @Time : 2022/5/10 11:17
# @Author : qingquan
# @File : request_demo.py
import json
import requests


url = "http://localhost:8000/digital_man"
post_dict = {"server": "好的，正在为您办理XXX业务，业务套餐为XXX元xxx分钟包含XX兆流量，请稍候", "client": "没听见"}

r1 = requests.post(url, data=json.dumps(post_dict))
print(r1.text)