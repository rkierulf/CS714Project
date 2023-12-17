import os
import imageio

test_freq = 20
def make_gif(pathname, scheme):
    images = []
    for val in range(len(os.listdir(pathname))-1):
        if val == 0: continue
        images.append(imageio.v2.imread(str(pathname) + '/{:03d}.png'.format(val*test_freq)))
    imageio.mimsave(str(os.getcwd()) + '/' + str(pathname) + '/training_ ' + str(scheme) + '.gif', images, format='GIF', fps=4)