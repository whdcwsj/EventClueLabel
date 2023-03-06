from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import logging
import json

from transformers import set_seed
from DataSource import DataSource
from event_classification.project_classification_refined import interface_classification
from event_clue_label.predict.predicter import interface_event_clue_predict
from event_extraction.inference import Inference
from event_extraction.arguments import get_args
from event_clue_infer.infer_event_clue import InferEventClue
from event_clue_infer.config import Config


# 定义接收数据的结构
class Item(BaseModel):
    id: str
    content: str
    level: int = 3
    is_probability_distributions: int = 1


class LabelItem(BaseModel):                                                                                                       
    id: str                                                                                                                  
    eventName: str                                                                                                           
    primaryClassification: str 


class ExtractionItem(BaseModel):
    txt_file_path: str


class ClueInferItem(BaseModel):
    id: str
    eventName: str
    action: str
    mainBody: str
    location: str
    time: str


class IdItem(BaseModel):
    id: str

DataSource = DataSource()
app=FastAPI()
logger = logging.getLogger()

args = get_args()
args.use_linear_bias = True  # 不要改动这个
args.use_copy = True  # 不要改动这个
args.use_gpu = False  # 用来控制是否使用gpu
args.gpu_device = 0  # 如果使用gpu，可以指定gpu的设备号
set_seed(args.seed)
inference = Inference(args)


@app.get("/")
def read_root():
    return {"hello": "world"}


@app.post('/eventClassify/predict')
async def event_classify(item: Item):
    text = item.content
    label, probability, classify_no = interface_classification(text, final_ckpt_path=r"event_classification/checkpoint", device="cpu")

    model_result = {"id":item.id, "classify_no":classify_no, "label":label, "probability_distributions": str(probability)}
    return model_result



@app.post('/eventCluelabel/predict')
async def event_cluelabel(item: LabelItem):
    eventName= item.eventName
    primaryClassification = item.primaryClassification

    scale, strength, score = interface_event_clue_predict(eventName, primaryClassification)

    model_result = {"id": item.id, "scale ": scale,
                    "strength": strength, "score": str(score)}
    return model_result


@app.post('/eventExtraction/predict')
async def event_extraction(item:ExtractionItem):
    #args = get_args()                                                                                      
    #args.use_linear_bias = True  # 不要改动这个                                                            
    #args.use_copy = True  # 不要改动这个                                                                   
    #args.use_gpu = False  # 用来控制是否使用gpu                                                             
    #args.gpu_device = 0  # 如果使用gpu，可以指定gpu的设备号                                                

    documents_path = item.txt_file_path
    print("lujinng:::::::::", documents_path)
    # documents_path = r"/app/biaozhu/annotation/data/中国第10批赴马里维和部队出征.txt"                       
    result = inference.inference(documents_path)
    return result


@app.post('/eventClueInfer/predict')
async def event_clue_infer(item:ClueInferItem):
    data_dict = {
        "id": item.id,
        "eventName": item.eventName,
        "mainBody": item.mainBody,
        "location": item.location,
        "action": item.action,
        "time": item.time
    }
    config = Config(is_infer=True)
    infer_event_clue = InferEventClue(config=config)
    result = infer_event_clue.infer(data_dict)
    return result


@app.post("/eventAll/predict")
async def event_all(item: IdItem):
    annotationId = item.id
    DataSource.is_connected()                                                                               
    conn = DataSource.conn                                                                                  
    cursor = conn.cursor()                                                                                 
    cursor.execute("select t.title as title,convert(t.content using utf8mb4) as content, t.type as type, t.sjsj as sjsj, t.txt_file_name as txt_file_name from sync_annotation_data_list t where id="+str(annotationId))                                                  
    result0 = cursor.fetchall()
    res_len = len(result0)                                                                                  
    modl_result = {}                                                                                     
    if res_len == 0:                                                                                        
        print("数据为空")                                                                                  
        model_result['status'] = '-1'                                                                      
        model_result['message'] = '数据为空'                                                               
        return model_result       
    #print(result)
    # conn.close()
    else:
        print(result0)
        # result = result[0]
        documents_path =result0[0][4]

        # print("documents_path::::", documents_path)
        text = result0[0][1]
        # documents_path = r"/app/biaozhu/annotation/data/中国第10批赴马里维和部队出征.txt"
        result = inference.inference(documents_path)
        label, probability, classify_no = interface_classification(text, final_ckpt_path=r"event_classification/checkpoint", device="cpu")                                                                                                  
        result = result[0]                                                                                                 
        result['event_list'][0]["event_attr"]["event_type"]=classify_no
        primaryClassification = classify_no[0]
        scale, strength, score = interface_event_clue_predict(text, primaryClassification)
        result['event_list'][0]['event_mark'] = score
        result['event_list'][0]['event_strength']=strength
        result['event_list'][0]['event_scale']=scale
        main = result['event_list'][0]["event_attr"]
        mainBody = main['mainBody'][0] if len(main['location'])> 1 else ""
        location = main['location'][0] if len(main['location'])>1 else ""
        action = main['action'][0] if len(main['action'])>1 else ""
        time = main['time'][0] if len(main['time'])>1 else ""

        data_dict = {                                                                                          
            "id": item.id,                                                                                     
            "eventName": text,                                                                       
            "mainBody": mainBody,                                                                         
            "location": location,                                                                         
            "action": action,                                                                             
            "time": time                                                                                  
        } 
        print(data_dict)
        config = Config(is_infer=True)                                                                         
        infer_event_clue = InferEventClue(config=config)                                                       
        clue_result = infer_event_clue.infer(data_dict)  
        result['event_list'][0]['event_area'] =clue_result['area']
        result['event_list'][0]['event_purpose'] = clue_result['purpose']
        result['event_list'][0]['event_actionStyle'] = clue_result['actionStyle']
        result['event_list'][0]['event_opportunity'] = clue_result['opportunity']
        result['status'] = '1'                                                                         
        result['message'] = '成功'
    conn.close()
    return result




if __name__ == '__main__':
    uvicorn.run(app='main:app',
                host="0.0.0.0",
                port=12456,
                reload = True,
                debug = True)
