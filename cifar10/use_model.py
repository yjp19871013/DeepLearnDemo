from keras.models import load_model
from PIL import Image
import numpy as np

type_list = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "狐狸", "马", "船", "卡车"]

pic_path = '3.jpg'
model_path = "model/cifar.h5"

im = Image.open(pic_path)
im = np.array(im).reshape(1, 32, 32, 3)
model = load_model(model_path)

result = model.predict(im)
may_be = np.argmax(result)
print(type_list[may_be])
