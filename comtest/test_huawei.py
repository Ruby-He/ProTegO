# -*- coding: utf-8 -*-
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkocr.v1 import *
import base64
import os,sys
from nltk.metrics import edit_distance
from utils import Logger

ak = "xx"
sk = "xx"

def get_file_content(filePath):
    with open(filePath, 'rb') as image_file:
        return base64.b64encode(image_file.read())
 
def main(imagefiles):
    credentials = BasicCredentials(ak, sk) 
    
    client = OcrClient.new_builder() \
        .with_credentials(credentials) \
        .with_region(OcrRegion.value_of("cn-east-3")) \
        .build()

    fcnt, ED_sum = 0, 0
    try:
        for imagefile in imagefiles:
            encoded_string = get_file_content(imagefile)
            # with open(file, "rb") as image_file:
            #     encoded_string = base64.b64encode(image_file.read())
            request = RecognizeGeneralTextRequest()
            request.body = GeneralTextRequestBody(image=encoded_string)
            response = client.recognize_general_text(request)
            data = response.result.to_dict()
            words = [word['words'] for word in data['words_block_list']]
            content = "".join(words)

            imagename = os.path.basename(imagefile)
            true_label = imagename.split("_")[1]
            if true_label != content:
                fcnt += 1
                ED = edit_distance(true_label, content)
            else:
                ED = 0
            print("label:{} ---> result:{}".format(true_label, content))
            ED_sum += ED
            
    except exceptions.ClientRequestException as e:
        print(e.status_code)
        print(e.request_id)
        print(e.error_code)
        print(e.error_msg)
    if fcnt != 0:
        score = {"DSR":fcnt/len(imagefiles), "ED":ED_sum/fcnt}
    else:
        score = {"DSR":0, "ED":0}
    return score

if __name__ == "__main__":
    log_file= os.path.join('./res-comtest/huawei', 'up5a.log')
    sys.stdout = Logger(log_file)

    # prepare your own test data
    file_path = "xx/adv+"
    # file_path = "xx/adv-"

    # get file paths
    imagefiles = [os.path.join(file_path,n) for n in os.listdir(file_path)]
    imagefiles = imagefiles[:1]
    score = main(imagefiles)
    print(score)
