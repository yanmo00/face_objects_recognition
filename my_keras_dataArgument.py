from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os

# 定义数据增强器
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(1, 1.4),
    rescale=1 / 255,
    shear_range=0.8,
    zoom_range=0.8,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 设置路径
input_folder = 'D:/resource/Note/Obsidian/picture'
output_folder = 'D:/resource/Note/Obsidian/picture_enhanced'

# 创建输出目录（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中所有图片文件
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = [f for f in os.listdir(input_folder)
               if os.path.splitext(f)[1].lower() in valid_extensions]

# 处理每张图片
total_generated = 0
for index, img_file in enumerate(image_files):
    if index < 210:  # 跳过前210张
        continue

    try:
        print(f"\n[{index + 1}/{len(image_files)}] 正在处理 {img_file}...")  # 新增进度显示

        img_path = os.path.join(input_folder, img_file)
        print("1. 正在加载图片...")
        img = load_img(img_path)

        print("2. 正在转换数组...")
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        print("3. 开始生成增强图片...")
        i = 0
        for batch in datagen.flow(x,
                                  batch_size=1,
                                  save_to_dir=output_folder,
                                  save_prefix=f'aug_{os.path.splitext(img_file)[0]}',
                                  save_format='jpeg'):
            i += 1
            print(f"  已生成 {i}/7", end='\r')  # 实时显示生成进度
            if i == 7:
                break
        total_generated += 7
        print(f"√ 完成 {img_file} 的处理")  # 新增完成标记

    except Exception as e:
        print(f"× 处理 {img_file} 时出错: {str(e)}")
