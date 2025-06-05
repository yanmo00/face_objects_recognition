face_recognition_facenet
├─main.py  主脚本，实现对单张图片的人脸检测和识别
├─get_database_faces_feature.py  获取后台数据库照片中的人脸特征向量
├─README.txt
├─data
│  ├─faces_feature_database.pkl      后台数据库人脸特征向量pkl文件
│  ├─database_images                     后台人脸图像数据库
│  ├─examples_images_results         图片识别结果文件
│  └─examples_images                     测试图片
├─models
│  ├─detect_face.py                         人脸检测及人脸部位图像提取
│  ├─inception_resnet_v1.py            人脸特征向量提取主干网络
│  ├─face_alignment.py                   人脸对齐操作
│  └─mtcnn.py                                 MTCNN源码，包含PNet/RNet/ONet
└─weights
   ├─facenet_inception_resnetv1.pt     人脸特征向量提取主干网络的权重文件（已经过预训练）
   ├─onet.pt                                        ONet的权重文件（已经过预训练）
   ├─pnet.pt                                        PNet的权重文件（已经过预训练）
   └─rnet.pt                                         RNet的权重文件（已经过预训练）

