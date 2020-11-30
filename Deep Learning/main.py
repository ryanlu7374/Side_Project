import argument_parser
import data_process
import model_selector
import os
import numpy as np
import pandas as pd
from keras.models import load_model
from pathlib import Path
import result

args = argument_parser.argumentparser()

command=(args.command)
if(args.IsRGB == 'True' or args.IsTestRGB == 'True'):
    GrayorRGB=True
elif(args.IsRGB == 'False' or args.IsTestRGB == 'False'):
    GrayorRGB=False
img_size=int(args.ImageSize)
valid_pc=float(args.TestRate)
traindata=args.TrainPath
modelname=args.Model
optimizername=args.Optimizer
learningrate=float(args.LearningRate)
decay=float(args.Decay)
epochs=int(args.Epochs)
testdata=args.TestPath
modelpath=args.ModelPath
GpuNum=2
if(args.IsLabel == 'True'):
    islabel=True
elif(args.IsLabel == 'False'):
    islabel=False
if(args.command =='train'):
    import callback

    pathlist = traindata.split('\\')
    try:
        pathlist.index('Train')
        rootPath = '\\'.join(pathlist[:pathlist.index('Train')])
        newPath  = rootPath + r'\saveLog\trainLog.txt'
    except ValueError:
        try:
            pathlist.index('train')
            rootPath = '\\'.join(pathlist[:pathlist.index('train')])
            newPath  = rootPath + r'\saveLog\trainLog.txt'
        except:
            print('Error Train Path')
    
elif(args.command =='test'):
    import callback

    pathlist = testdata.split('\\')
    try:
        pathlist.index('Test')
        rootPath = '\\'.join(pathlist[:pathlist.index('Test')])
        newPath  = rootPath + r'\saveLog\testLog.txt'
    except ValueError:
        try:
            pathlist.index('test')
            rootPath = '\\'.join(pathlist[:pathlist.index('test')])
            newPath  = rootPath + r'\saveLog\testLog.txt'
        except:
            print('Error Test Path')
    
else:
    print("TestPath Pass Null Error")

writer = result.Record(newPath)
writer.writefile(inf_list=["SamplePath:{0}\n".format(newPath)] ,inf_type='Info')    

if(command=='train'):
    print("============train _start============")
    print("command =", command)
    print("GrayorRGB =", GrayorRGB)
    print("valid_pc =", valid_pc)
    print("traindata =", traindata)
    print("modelname =", modelname)
    print("optimizername =", optimizername)
    print("learningrate =", learningrate)
    print("decay =", decay)
    print("epochs =", epochs)
    
    img_size=model_selector.SelectInputSize(ModelName=modelname)
    print("img_size =", img_size)

    x_train_paths, x_valid, y_train_OnH, y_valid_OnH, class_weight = data_process.load_data(
        GrayorRGB=GrayorRGB, img_size=img_size, valid_pc=valid_pc , samplepath = traindata, label=True)

    model, singlemodel = model_selector.SelectModel(ModelName=modelname, ImgRow=x_valid.shape[1], ImgCol=x_valid.shape[2], 
                                                    ImgChannel=x_valid.shape[3], Classes=y_valid_OnH.shape[1], GpuNum=GpuNum)

    print('model =', model.name)
    print(model.summary())
    
    optimizer = model_selector.SelectOptimizer(OptimizerName=optimizername, LearningRate=learningrate, Decay=decay)
    print('optimizer =', optimizer)
    
    savemodel ,filename = callback.SaveModel(SingleModel=singlemodel, SamplePath=traindata, ModelName=modelname, OptimizerName=optimizername, Monitor='val_acc')
    print('savemodel = ', savemodel)
    print('filename = ', filename)
    
    callbacks= [callback.TrainingLog(newPath), savemodel]

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    train_history=model.fit_generator(data_process.yield_generater(x_train_paths, y_train_OnH, 6, GrayorRGB, img_size),
                                      steps_per_epoch=len(x_train_paths)/6, epochs=epochs, callbacks=callbacks,
                                      max_queue_size=1, validation_data=(x_valid, y_valid_OnH), 
                                      workers=1, class_weight=class_weight)    
    
elif(command=='test'):
    writer.writefile(inf_list=['============test_start============'],inf_type='Info')

    loaded_model=load_model(modelpath)
    writer.writefile(inf_list=["load_test_data done"] ,inf_type='Info')

    img_size=loaded_model.get_input_shape_at(0)[1]
    writer.writefile(inf_list=["img_size = {0}".format(img_size)] ,inf_type='Info')
    
    x_test, x_test_id, y_test, y_test_OnH = data_process.load_test_data(GrayorRGB=GrayorRGB, img_size=img_size, samplepath = testdata, label=islabel)
    writer.writefile(inf_list=["x_test = {0}".format(x_test.shape)] ,inf_type='Info')
    writer.writefile(inf_list=["y_test = {0}".format(y_test.shape)] ,inf_type='Info')
    writer.writefile(inf_list=["y_test_OnH = {0}".format(y_test_OnH.shape)] ,inf_type='Info')
    
    file=[]
    for file_name in x_test_id:
        fn=Path(file_name)
        file.append(fn.name.lower())    

    prediction=loaded_model.predict(x_test)

    TrnNamelist = result.create_folder(rootPath)    
    y_test_pred = prediction.argmax(axis=-1).astype(np.int32)
    output_df=pd.DataFrame(columns=["file_name","label","test_pred"])
    output_df["file_name"]=file
    if (islabel==True):
        output_df["label"]=y_test.astype(np.int32)
    output_df["test_pred"]= y_test_pred
    output_df.to_csv(rootPath+r'\saveLog\output.csv',sep=',')
    writer.writefile(inf_list=["test fuction done"] ,inf_type='Info')
    
    if(islabel):
        scores = loaded_model.evaluate(x_test,y_test_OnH)
        writer.writefile(inf_list=["scores = {0}".format(scores)] ,inf_type='Info')
        
        conf=pd.crosstab(y_test,y_test_pred,rownames=['label'],colnames=['predict'])
        writer.writefile(inf_list=["conf = {0}".format(conf)] ,inf_type='Info')