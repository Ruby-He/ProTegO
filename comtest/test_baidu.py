#!/usr/bin/python3
import os,sys
import time
from aip import AipOcr
from nltk.metrics import edit_distance
from utils import Logger

APP_ID = 'xx'
API_KEY = 'xx'
SECRET_KEY = 'xx'
request_interval = 1

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def get_score(jsons):
    fcnt, ED_sum = 0, 0
    for json in jsons:
        fcnt +=  0 if json["label"] == json["words_result"][0]["words"] else 1
        ED_sum += edit_distance(json["label"], json["words_result"][0]["words"])
        if fcnt != 0:
            score = {"DSR":fcnt/len(jsons), "ED":ED_sum/fcnt}
        else:
            score = {"DSR":fcnt/len(jsons), "ED":0}
    return score


if __name__ == "__main__":
    log_file= os.path.join('./res-comtest/baidu', 'up5a.log')
    sys.stdout = Logger(log_file)
    # create client
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

    # prepare your own test data
    data_path = "xx/adv+"
    # data_path = "xx/adv-"

    imagefiles = [os.path.join(data_path,n) for n in os.listdir(data_path)]
    imagefiles = imagefiles[:100]

    # upload to baidu ocr
    result_jsons = []
    for imagefile in imagefiles:
        b_img = get_file_content(imagefile)
        answer = client.basicGeneral(b_img)
        imagename = os.path.basename(imagefile)
        answer["label"] = imagename.split("_")[1]
        try:
            print("label:{} ---> result:{}".format(
                answer["label"],answer["words_result"][0]["words"]
                ))
        except:
            answer["words_result"].append({'words':""})
        print(answer)
        result_jsons.append(answer)
        time.sleep(request_interval)
    
    # calc score 
    score = get_score(result_jsons)
    print(score)


'''
basicGeneral: 通用文字识别（标准版）
return json:
{
    "words_result": [
        {
            "words": "firm"
        }
    ],
    "words_result_num": 1,
    "log_id": 1589115807642170340
}

not found words return json:
return json:
{
    "words_result": [],
    "words_result_num": 0,
    "log_id": 1589115807642170340
}
'''