import tkinter as tk
from PIL import Image, ImageTk
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt 
from models.VAE import VariationalAutoencoder
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

    
# In[ ]:


# run params
section = 'vae'
run_id = '0002'
data_name = 'faces'
RUN_FOLDER = 'run/{}/'.format(section)
RUN_FOLDER += '_'.join([run_id, data_name])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode =  'load' #'load' #


DATA_FOLDER = './youmu2/'


# ## data

# In[ ]:


INPUT_DIM = (128,128,3)
BATCH_SIZE = 32

filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.png')))
print(filenames)
NUM_IMAGES = len(filenames)


# In[ ]:


data_gen = ImageDataGenerator(rescale=1./255)

data_flow = data_gen.flow_from_directory(DATA_FOLDER
                                         , target_size = INPUT_DIM[:2]
                                         , batch_size = BATCH_SIZE
                                         , shuffle = True
                                         , class_mode = 'input'
                                         , subset = "training"
                                            )


# ## architecture

# In[ ]:


vae = VariationalAutoencoder(
                input_dim = INPUT_DIM
                , encoder_conv_filters=[32,64,64, 64]
                , encoder_conv_kernel_size=[3,3,3,3]
                , encoder_conv_strides=[2,2,2,2]
                , decoder_conv_t_filters=[64,64,32,3]
                , decoder_conv_t_kernel_size=[3,3,3,3]
                , decoder_conv_t_strides=[2,2,2,2]
                , z_dim=25
                , use_batch_norm=True
                , use_dropout=True)

if mode == 'build':
    vae.save(RUN_FOLDER)
else:
    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/default.h5'))


# In[ ]:


#vae.encoder.summary()


# In[ ]:


#vae.decoder.summary()


# ## training

# In[ ]:


LEARNING_RATE = 0.0004
R_LOSS_FACTOR = 10000
EPOCHS = 0
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0


# In[ ]:


vae.compile(LEARNING_RATE, R_LOSS_FACTOR)


# In[ ]:


vae.train_with_generator(     
    data_flow
    , epochs = EPOCHS
    , steps_per_epoch = NUM_IMAGES / BATCH_SIZE
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , initial_epoch = INITIAL_EPOCH
)

print('VAE ready')


img_size = 200
div_size = 200

window = tk.Tk()
window.title('window')


def define_layout(obj, cols=1, rows=1):
    
    def method(trg, col, row):
        
        for c in range(cols):    
            trg.columnconfigure(c, weight=1)
        for r in range(rows):
            trg.rowconfigure(r, weight=1)

    if type(obj)==list:        
        [ method(trg, cols, rows) for trg in obj ]
    else:
        trg = obj
        method(trg, cols, rows)

DIM = 25
EACH_ROW =5
bars = []


    
    

def indexOf(wid,bars):
    i=0
    for w in bars:
        if(w==wid):
            return i
        i+=1
    return -1


def generateImage():
    fig = plt.figure(figsize=(3,3))
    reconst = vae.decoder.predict(np.array(znew))
    fig.subplots_adjust(hspace=0.4,wspace=0.4)
    for i in range(1):
              ax = fig.add_subplot(1,1,1)
              ax.imshow(reconst[i,:,:,:])
              ax.axis('off')
    path_resize = f'result2.png'
    plt.savefig(path_resize)
    plt.close('all')

def changeImage():
    
    im = Image.open('result2.png')
    imTK = ImageTk.PhotoImage(im.resize( (200, 200) ) )
    lbl_2.configure(image=imTK)
    lbl_2.image = imTK
    

znew = np.random.normal(0,0,size=(1,DIM))
def bar_event(self):
    znew[0][indexOf(self.widget,bars)]= self.widget.get()
    vectorDisplay.config(text =str(znew))
    generateImage()
    changeImage()


def reset_event(self):
    for i in range( DIM):
            znew[0][i] = 0
            
    vectorDisplay.config(text =str(znew))
    for sc in bars:
        sc.set(0)
    generateImage()
    changeImage()

def generateBar(dim,grid=None):
    for i in range(dim):
        temp = tk.Scale(grid, from_=-4,to=4,label=f'{i}',resolution=0.1,orient="horizontal", bg='#ed689d', fg='#ffebf3',troughcolor='#fce1ec')
        temp.bind("<ButtonRelease-1>",bar_event)
    
        temp.grid(column=int(i/EACH_ROW),row=i%EACH_ROW)
        bars.append(temp)

  

#window.geometry('600x800')
lbl_1 = tk.Label(window, text='Hello World', bg='yellow', fg='#263238', font=('Arial', 12))
div1 = tk.Frame(window,  width=img_size*3 , height=img_size , bg='#9c8e8e')

div1.columnconfigure(1, weight=5)

div2 = tk.Frame(window,  width=img_size*3 , height=40, bg='#e3b8b6')
div3 = tk.Frame(window,  width=img_size*3 , height=670 , bg='#8a8a8a')

div1.grid(column=0, row=0)
div2.grid(column=0, row=1)
div3.grid(column=0, row=2)

im = Image.open('./images/result.png')
imgTk = ImageTk.PhotoImage(im.resize( (200, 200) ) )
lbl_2 = tk.Label(div1, image=imgTk)
lbl_2.grid(column=0, row=0)
lbl_2.image = imgTk

define_layout(window, cols=1, rows=3)
define_layout([div1, div2, div3])
generateBar(25,grid = div3)


vectorDisplay = tk.Label(div2, text=znew, font=('Arial', 12))
vectorDisplay.bind("<Button-1>",  reset_event)
vectorDisplay.grid(column=0,row=0)
reconst = vae.decoder.predict(np.array(znew))
window.mainloop()













