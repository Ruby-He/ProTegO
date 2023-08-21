# -*- coding: utf-8 -*-
import os,sys
import json
from typing import List

from alibabacloud_tea_openapi.client import Client as OpenApiClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models
from nltk.metrics import edit_distance
from utils import Logger

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()
    
class Sample:
    def __init__(self):
        pass

    @staticmethod
    def create_client(
        access_key_id: str,
        access_key_secret: str,
    ) -> OpenApiClient:
        """
        使用AK&SK初始化账号Client
        @param access_key_id:
        @param access_key_secret:
        @return: Client
        @throws Exception
        """
        config = open_api_models.Config(
            # 必填，您的 AccessKey ID,
            access_key_id=access_key_id,
            # 必填，您的 AccessKey Secret,
            access_key_secret=access_key_secret
        )
        # 访问的域名
        config.endpoint = f'ocr-api.cn-hangzhou.aliyuncs.com'
        return OpenApiClient(config)

    @staticmethod
    def create_api_info() -> open_api_models.Params:
        """
        API 相关
        @param path: params
        @return: OpenApi.Params
        """
        params = open_api_models.Params(
            # 接口名称,
            action='RecognizeGeneral',
            # 接口版本,
            version='2021-07-07',
            # 接口协议,
            protocol='HTTPS',
            # 接口 HTTP 方法,
            method='POST',
            auth_type='AK',
            style='V3',
            # 接口 PATH,
            pathname=f'/',
            # 接口请求体内容格式,
            req_body_type='json',
            # 接口响应体内容格式,
            body_type='json'
        )
        return params

    @staticmethod
    def main(
        imagefiles: List[str],
    ) -> None:
        # 工程代码泄露可能会导致AccessKey泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考，建议使用更安全的 STS 方式，更多鉴权访问方式请参见：https://help.aliyun.com/document_detail/378659.html
        client = Sample.create_client('xx', 'xx')
        params = Sample.create_api_info()
        # runtime options
        runtime = util_models.RuntimeOptions()

        fcnt, ED_sum = 0, 0
        for imagefile in imagefiles:
            img = get_file_content(imagefile)
            request = open_api_models.OpenApiRequest(stream=img)
            # 复制代码运行请自行打印 API 的返回值
            # 返回值为 Map 类型，可从 Map 中获得三类数据：响应体 body、响应头 headers、HTTP 返回的状态码 statusCode
            result = client.call_api(params, request, runtime)
            data = json.loads(result['body']['Data'])
            content = data['content'].strip()
            # print(f"file: {file}")
            # print(content)
            imagename = os.path.basename(imagefile)
            true_label = imagename.split("_")[1]
            if true_label != content:
                fcnt += 1
                ED = edit_distance(true_label, content)
            else:
                ED = 0
            print("label:{} ---> result:{}".format(true_label, content))
            ED_sum += ED 
        
        if fcnt != 0: 
            score = {"DSR":fcnt/len(imagefiles), "ED":ED_sum/fcnt}
        else:
            score = {"DSR":0, "ED":0}
        return score

if __name__ == '__main__':
    log_file= os.path.join('./res-comtest/ali', 'up5a.log')
    sys.stdout = Logger(log_file)

    # prepare your own test data
    file_path = "xx/adv+"
    # file_path = "xx/adv-"

    # get file paths
    imagefiles = [os.path.join(file_path,n) for n in os.listdir(file_path)]
    imagefiles = imagefiles[:100]
    score = Sample.main(imagefiles)
    print(score)