from keras.utils import multi_gpu_model
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

def SelectModel(ModelName='ResNet50', ImgRow=224, ImgCol=224, ImgChannel=3, Classes=2, GpuNum=2):
    
    training_model = ModelName
    
    """ select DL model """
    if training_model.lower() == 'resnet50':
        from keras.applications.resnet50 import ResNet50
        # 使用全ResNet50，不加層
        model = ResNet50(weights=None, include_top=True,
                         input_shape=(ImgRow, ImgCol, ImgChannel), classes = Classes)
        for layer in model.layers:
            layer.trainable = True

    elif training_model.lower() == 'vgg16':
        from keras.applications.vgg16 import VGG16

        model = VGG16(weights=None, include_top=False,
                         input_shape=(ImgRow, ImgCol, ImgChannel), classes = Classes)
        x = model.output
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(Classes, activation='softmax')(x)
        model = Model(inputs=model.input, outputs=predictions)
        for layer in model.layers:
            layer.trainable = True

    elif training_model.lower() == 'vgg19':
        from keras.applications.vgg19 import VGG19
        
        model = VGG19(weights=None, include_top=False,
                         input_shape=(ImgRow, ImgCol, ImgChannel), classes = Classes)
        x = model.output
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(Classes, activation='softmax')(x)
        model = Model(inputs=model.input, outputs=predictions)
        for layer in model.layers:
            layer.trainable = True

    elif training_model.lower() == 'inception_v3' :
        from keras.applications.inception_v3 import InceptionV3
        model = InceptionV3(weights=None, include_top=True,
                         input_shape=(ImgRow, ImgCol, ImgChannel), classes = Classes)
        for layer in model.layers:
            layer.trainable = True

    elif training_model.lower() == 'inception_v4' :
        import inception_v4
        
        model = inception_v4.create_model(dropout_prob=0.2, weights='imagenet', include_top=True)
        for layer in model.layers:
            layer.trainable = False
        x = model.output
        predictions = Dense(Classes, activation='softmax')(x)
        model = Model(inputs=model.input, outputs=predictions)
    
    """ enable multiGPU """
    try:
        Model = multi_gpu_model(model, gpus=GpuNum) 
        print("Training using multiple GPUs..")
    except ValueError:
        Model = model
        print("Training using single GPU or CPU..")
    
    SingleModel=model
    return Model, SingleModel


def SelectInputSize(ModelName='ResNet50'):
    
    training_model = ModelName

    if training_model.lower() == 'resnet50' and 'vgg16' and 'vgg19':
        img_size=224
    
    elif training_model.lower() == 'inception_v3' and 'inception_v4':
        img_size=299
        
    return img_size

def SelectOptimizer(OptimizerName='Adam', LearningRate=0.001, Decay=0.001):
    
    Optimizer = OptimizerName

    if Optimizer == 'SGD':
        from keras.optimizers import SGD
        return SGD(lr=LearningRate, decay=Decay)

    elif Optimizer == 'RMSprop':
        from keras.optimizers import RMSprop
        return RMSprop(lr=LearningRate, decay=Decay)

    elif Optimizer == 'Adagrad':
        from keras.optimizers import Adagrad
        return Adagrad(lr=LearningRate, decay=Decay)

    elif Optimizer == 'Adadelta':
        from keras.optimizers import Adadelta
        return Adadelta(lr=LearningRate, decay=Decay)

    elif Optimizer == 'Adam':
        from keras.optimizers import Adam
        return Adam(lr=LearningRate, decay=Decay)
        
    elif Optimizer == 'Adamax':
        from keras.optimizers import Adamax
        return Adamax(lr=LearningRate, decay=Decay)
    
    elif Optimizer == 'Nadam':
        from keras.optimizers import Nadam
        return Nadam(lr=LearningRate, schedule_decay=Decay)