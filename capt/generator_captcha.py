#coding:utf-8
###验证码产生器
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

#扩展验证码创建方法，可定制噪点大小，数量
class ImageCaptcha2(ImageCaptcha):
    def __init__(self, width=160, height=60, fonts=None, font_sizes=None,
                 noise_dots_size=3, noise_dots_count=30):
        super(ImageCaptcha2, self).__init__(width=160, height=60, fonts=None, font_sizes=None)
        self.noise_dots_size=noise_dots_size
        self.noise_dots_count = noise_dots_count

    @staticmethod
    def create_noise_dots(image, color, width=3, number=30):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image



number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ALL_char_set = number + alphabet + ALPHABET


class



class Generator(object):
    def __init__(self, captcha_size=4, char_set=ALL_char_set, save_path='D:\\workspace\\dl\\data\\',
                 img_width=60, img_height=160):
        self.generate = ImageCaptcha(20,20).generate
        self.captcha_size = captcha_size
        self.char_set = char_set
        self.save_path = save_path
        self.txt = None
        self.img = None

    def generate_captcha(self, txt=None):
        self.make_txt(txt)
        self.make_img()

    def make_txt(self, txt=None):
        self.txt = txt or ''.join([random.choice(self.char_set) for i in range(self.captcha_size)])

    def make_img(self):
        img = self.generate(self.txt)
        self.img = img

    def show(self):
        img_data = np.array(Image.open(self.img))
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, self.txt, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(img_data)
        plt.show()


if __name__ == '__main__':
    g = Generator(1)
    g.generate_captcha('X')
    g.show()









