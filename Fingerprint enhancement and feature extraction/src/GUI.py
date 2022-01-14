#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
from functions import *
from PIL import ImageTk, Image
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)


# In[77]:

def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))


    while True:
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open)
 
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        if cv2.countNonZero(img)==0:
            break
    return skel

temp = np.array(0)
temp2 = np.array(0)
temp4 = np.array(0)
temp7 = np.array(0)

def open_img():
    
    global temp
    x = openfilename()
 
    temp = loadImage(x)
    img = Image.fromarray(np.uint8(temp*255))
    
    plot(img, 2, 2)
    
    img = img.resize((250, 250), Image.ANTIALIAS)
 
    img = ImageTk.PhotoImage(img)
  
    panel = Label(root, image = img)
     
    panel.image = img
    panel.grid(row = 2)
    
def enhance_img():
    global temp2
    global temp4
    temp4 = weakLightEnhancement(temp)
    temp2 = Image.fromarray(temp4)
    plot(temp2, 30, 2)
    temp3 = temp2.resize((250, 250), Image.ANTIALIAS)
    temp3 = ImageTk.PhotoImage(temp3)

    panel = Label(root, image = temp3)   
    panel.image = temp3
    panel.grid(row = 30)
    
def save_img():
    global temp2
    temp2.save("output.png")
    
def openfilename():
    filename = filedialog.askopenfilename(title ='"pen')
    return filename

def plot(im, x, y):
    fig = Figure(figsize = (6, 2), dpi = 100)
    
    # adding the subplot
    plot1 = fig.add_subplot(131)
    plot2 = fig.add_subplot(132)
    plot3 = fig.add_subplot(133)
    
    histogram = im.histogram()  
    
    Red=histogram[0:256]      # indicates Red
    Green=histogram[256:512]  # indicated Green
    Blue=histogram[512:768]   # indicates Blue
    
    plot1.hist(np.array(Red).ravel(), 256, (0, 256))
    plot2.hist(np.array(Green).ravel(), 256, (0, 256))
    plot3.hist(np.array(Blue).ravel(), 256, (0, 256))
    
    plot1.title.set_text('Red')
    plot2.title.set_text('Green')
    plot3.title.set_text('Blue')
    
  
    plot1.plot()
  
    canvas = FigureCanvasTkAgg(fig, master = root)  
    canvas.draw()
  
    canvas.get_tk_widget().grid(row = x, column = y)
    
def feature_extraction(img):
    total_feature_points = 0
    im = img.copy()
    for r in range(1,img.shape[0]-1):
        for c in range(1,img.shape[1]-1):
            sum = np.sum(img[r-1:r+2,c-1:c+2])
            if sum ==5 or sum == 7 or sum == 4:
                im[r][c]= 0
                total_feature_points +=1
            else:
                im[r][c]=255
    return im    
    
def extract_features():
    global temp4
    global temp7
    (row, col) = temp4.shape[0:2]
    im_m2 = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            im_m2[i, j] = sum(temp4[i, j]) * 0.33/255
    im_m2 = np.where(im_m2>0.6,1,0)
    im_m2 = skeletonize(im_m2.astype(np.uint8))
    im_m2 = feature_extraction(im_m2)
    
    
    rgb_im_m2 = cv2.merge((im_m2, im_m2, im_m2))
    
    temp5 = Image.fromarray(np.uint8(rgb_im_m2))
    temp7 = temp5
    temp6 = temp5.resize((250, 250), Image.ANTIALIAS)
    temp6 = ImageTk.PhotoImage(temp6)

    panel = Label(root, image = temp6)   
    panel.image = temp6
    panel.grid(row = 60)
    
def save_features():
    global temp7
    temp7.save("features.png")
    

root = Tk()
 
root.title("Image Loader")
 
root.geometry("1200x1200+500+500")
 
root.resizable(width = True, height = True)
 
btn = Button(root, text ='open image', command = open_img).grid(row = 1, columnspan = 4)

btn2 = Button(root, text ='enhance image', command = enhance_img).grid(row = 10, columnspan = 4)

btn3 = Button(root, text ='save image', command = save_img).grid(row = 40, columnspan = 4)

btn4 = Button(root, text ='extract features', command = extract_features).grid(row = 60, columnspan = 4)

btn5 = Button(root, text ='save features', command = save_features).grid(row = 80, columnspan = 4)

root.mainloop()

