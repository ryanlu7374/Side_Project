def argumentparser():
    import argparse

    # Parse command line arguments

    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect object class.')

    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'",
                        default='train')
    
    parser.add_argument('--IsRGB', required=False, default=True,
                        help='Gray or RGB image')

    parser.add_argument('--two2three', required=False, default=False,
                        help="Gray image，2D change into 3D")
    
    parser.add_argument('--ImageSize', required=False, default=224,
                        help='Image size for training')
    
    parser.add_argument('--TestRate', required=False, default=0.18,
                        help='According to the percentage, split the data set as the validation set')
    
    parser.add_argument('--TrainPath', required=False, 
                        default='D:/Contrel_Project/project/_DATA/sample_ASE/sample_ASE_20190214/train',
                        help='Folder path')
    
    parser.add_argument('--Model', required=False, 
                        default='ResNet50',
                        help='Select AI Model')

    parser.add_argument('--Optimizer', required=False, 
                        default='Adam',
                        help='Select Optimizer')

    parser.add_argument('--LearningRate', required=False, 
                        default=0.001,
                        help='LearningRate')
    
    parser.add_argument('--Decay', required=False, 
                        default=0.001,
                        help='Decay for LearningRate decrease dagree')
    
    parser.add_argument('--Epochs', required=False, 
                        default=3,
                        help='Epochs')
    
    parser.add_argument('--TestPath', required=False, 
                        help='Folder or File path')
    
    parser.add_argument('--ModelPath', required=False, 
                        help='model path')
    
    parser.add_argument('--IsLabel', required=False, 
                        default='False',
                        help='Does file have label?')

    parser.add_argument('--IsTestRGB', required=False, default=True,
                        help='Gray or RGB image')
    
    parser.add_argument('--ImgPath', required=False, type=list, action='store', dest='list',
                        help='Test Image Path List')

    args = parser.parse_args()

#     args = parser.parse_args(['--GrayorRGB', 'True', '--two2three', 'False',
#                               '--img_size', '224', '--valid_pc', '0.18',
#                               '--folderpath', 'C:/'])
#     parser.print_help()

    return args