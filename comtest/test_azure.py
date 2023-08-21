# need pip install azure-cognitiveservices-vision-computervision pillow
# 5 call per minute, 5k call per month!!!
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials


import os,sys,time
from nltk.metrics import edit_distance
from utils import Logger

'''
Authenticate
Authenticates your credentials and creates a client.
'''
subscription_key = "xx"
endpoint = "xx"
request_interval = 4

def main(imagefiles):
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    
    fcnt, ED_sum = 0, 0
    for imagefile in imagefiles:
        imagename = os.path.basename(imagefile)
        true_label = imagename.split("_")[1]
        # print(f"file: {file}")
        with open(imagefile, 'rb') as image_steam:
            read_response = computervision_client.recognize_printed_text_in_stream(image_steam, raw=True)
            time.sleep(request_interval)
        for region in read_response.output.regions:
            lines = region.lines
            for line in lines:
                line_text = " ".join([word.text for word in line.words])
            if true_label != line_text:
                fcnt += 1
                ED = edit_distance(true_label, line_text)
            else:
                ED = 0
            print("label:{} ---> result:{}".format(true_label, line_text))
            ED_sum += ED
    if fcnt != 0: 
        score = {"DSR":fcnt/len(imagefiles), "ED":ED_sum/fcnt}
    else:
        score = {"DSR":0, "ED":0}
    return score
        


if __name__ == "__main__":
    log_file= os.path.join('./res-comtest/azure', 'up5a.log')
    sys.stdout = Logger(log_file)

    # prepare your own test data
    file_path = "xx/adv+"
    # file_path = "xx/adv-"

    imagefiles = [os.path.join(file_path,n) for n in os.listdir(file_path)]
    score = main(imagefiles)
    print(score)