import tkinter as tk
import tkinter.messagebox
from pathlib import Path
from PIL import Image, ImageTk
import pre_processors
import controlled_degradations
import edge_detectors
import os

## GUI Structure:
# original image | preprocessors | preprocessed image | controlled degradations | degraded image
# | edge detectors | final image |
# export final-annotated side by side
# export process (original-preprocessed-degraded-final)
class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        # RBD-SLDI
        self.title("Robust Boundary Detection for Skin Lesions in Dermoscopic Images")
        self.minsize(1200, 800)
        # placeholder values to carry the imgpath of the currently targeted image
        self.currentimage = None

        # checker values
        self.nodegradechecker = False

        # custom close protocol
        self.protocol("WM_DELETE_WINDOW", self.rmtemp)

        # creating first GUI frame for image selector
        selectorframe = tk.Frame(self, borderwidth=5, relief="groove",
                                 width=150, height=200)
        selectorframe.grid(row=0, column=0, sticky="nsew", padx=10)

        # recursively add buttons for every image in selector frame
        tk.Label(selectorframe,text="Available Images:").grid(row=0, column=0)
        imagedirectory = Path("image-loader")
        iteration = 1
        for images in imagedirectory.iterdir():
            # button label name
            imagename = images.name
            # PIL thumbnail preprocessing and conversion
            pilimage = Image.open(str(images))
            pilimage = pilimage.resize((25,25))
            thumbnail = ImageTk.PhotoImage(pilimage)
            # button creation, passing imgpath
            item = tk.Button(selectorframe,image=thumbnail, text=imagename, compound = "left",
                      command=lambda imgpath=str(images) : self.imageselect(imgpath))
            item.grid(row=iteration, column=0, sticky="w")
            # ensure image is persistent
            item.image = thumbnail
            # ensure frame grows with each added image
            selectorframe.grid(rowspan=iteration)
            iteration += 1
    ## button command for selecting the image to process
    # takes image path string as an argument
    def imageselect(self, imgpath):
        ## temp logging
        print("Image Select Button Pushed")
        ##
        self.currentimage = imgpath
        originalimg = Image.open(imgpath)
        # resizing image to fit display
        originalimg.thumbnail((200,200))
        gui_img = ImageTk.PhotoImage(originalimg)
        origlabel = tk.Label(self, image=gui_img, text="Original Image:", compound="top")
        origlabel.grid(row=0, column=1, sticky="nsew")
        origlabel.image = gui_img
        # build the rest of the GUI frames and buttons
        self.completeGUI()
    ## function for building the rest of the GUI after an original image has been selected
    def completeGUI(self):
        ### additional frames for each processing type and their corresponding buttons
        ## preprocessor
        preprocessorframe = (tk.Frame(self, borderwidth=5, relief="groove",
                                     width=150, height=200))
        preprocessorframe.grid(row=0, column=2, sticky="nsew", padx=10)
        tk.Label(preprocessorframe, text="Preprocessing Tools:").grid(row=0, column=0)
        # Gaussian Smoothing
        gausssmooth = tk.Button(preprocessorframe, text="Gaussian Smoothing",
                                command=lambda: self.preprocess("Gaussian Smoothing"))
        gausssmooth.grid(row=1, column=0, sticky="nsew")
        # Median Filtering
        median = tk.Button(preprocessorframe, text="Median Filtering",
                                command=lambda: self.preprocess("Median Filtering"))
        median.grid(row=2, column=0, sticky="nsew")
        # Bilateral Filtering
        bilateral = tk.Button(preprocessorframe, text="Bilateral Filtering",
                           command=lambda: self.preprocess("Bilateral Filtering"))
        bilateral.grid(row=3, column=0, sticky="nsew")
        # Histogram Equalization
        histo = tk.Button(preprocessorframe, text="Histogram Equalization",
                           command=lambda: self.preprocess("Histogram Equalization"))
        histo.grid(row=4, column=0, sticky="nsew")
        # CLAHE Contrast Limited Adaptive Histogram Equalization
        clahe = tk.Button(preprocessorframe, text="Contrast Limited Adaptive Histogram Equalization",
                           command=lambda: self.preprocess("CLAHE"))
        clahe.grid(row=2, column=0, sticky="nsew")
        ## controlled degradations
        degradationframe = tk.Frame(self, borderwidth=5, relief="groove",
                                    width=150, height=200)
        degradationframe.grid(row=0, column=3, sticky="nsew", padx=10)
        tk.Label(degradationframe, text="Controlled Degradation Tools:").grid(row=0, column=0)
        # None
        none = tk.Button(degradationframe, text="None",
                               command=lambda: self.degrade("None"))
        none.grid(row=1, column=0, sticky="nsew")
        # Gaussian Noise
        gaussnoise = tk.Button(degradationframe, text="Gaussian Noise",
                                command=lambda: self.degrade("Gaussian Noise"))
        gaussnoise.grid(row=2, column=0, sticky="nsew")
        # Salt and Pepper Noise
        saltpepper = tk.Button(degradationframe, text="Salt and Pepper Noise",
                               command=lambda: self.degrade("Salt and Pepper Noise"))
        saltpepper.grid(row=3, column=0, sticky="nsew")
        # Blur
        blur = tk.Button(degradationframe, text="Blur",
                               command=lambda: self.degrade("Blur"))
        blur.grid(row=4, column=0, sticky="nsew")
        # Reduced Illumination
        debright = tk.Button(degradationframe, text="Reduce Illumination",
                         command=lambda: self.degrade("Reduce Illumination"))
        debright.grid(row=5, column=0, sticky="nsew")
        ## edge detectors
        edgeframe = tk.Frame(self, borderwidth=5, relief="groove",
                             width=150, height=200)
        edgeframe.grid(row=0, column=4, sticky="nsew", padx=10)
        tk.Label(edgeframe, text="Edge Detection Tools:").grid(row=0, column=0)
        # Sobel
        sobel = tk.Button(edgeframe, text="Sobel",
                             command=lambda: self.edgedet("Sobel"))
        sobel.grid(row=1, column=0, sticky="nsew")
        # Prewitt
        prewitt = tk.Button(edgeframe, text="Prewitt",
                          command=lambda: self.edgedet("Prewitt"))
        prewitt.grid(row=2, column=0, sticky="nsew")
        # Laplacian of Gaussian
        log = tk.Button(edgeframe, text="Laplacian of Gaussian",
                          command=lambda: self.edgedet("LoG"))
        log.grid(row=3, column=0, sticky="nsew")
        # Canny Edge Detection
        canny = tk.Button(edgeframe, text="Canny Edge Detection",
                          command=lambda: self.edgedet("Canny"))
        canny.grid(row=4, column=0, sticky="nsew")
    ## function to create preprocessed image and create it's associated
    def preprocess(self, type):
        ## temp logging
        print("Preprocess Button Pushed")
        ##
        # type selector
        if type == "Gaussian Smoothing":
            pre_processors.gaussSmooth(self.currentimage)
        elif type == "Median Filtering":
            pre_processors.medianFilter(self.currentimage)
        elif type == "Bilateral Filtering":
            pre_processors.bilateralFilter(self.currentimage)
        elif type == "Histogram Equalization":
            pre_processors.histogramEqual(self.currentimage)
        elif type == "CLAHE":
            pre_processors.CLAHE(self.currentimage)
        # open processed image
        pilimg = Image.open('preprocessed.png')
        # resizing image to fit display
        pilimg.thumbnail((200, 200))
        gui_img = ImageTk.PhotoImage(pilimg)
        preprocesslabel = tk.Label(self, image=gui_img, text=f"{type} Preprocessed Image:", compound="top")
        preprocesslabel.grid(row=1, column=2, sticky="nsew", padx=10)
        preprocesslabel.image = gui_img
    ## function to create degraded images from preprocessed image
    def degrade(self, type):
        ### temp logging
        print("Degrade Button Pushed")
        ###
        self.nodegradechecker = False
        if os.path.exists("preprocessed.png"):
            if type == "Gaussian Noise":
                controlled_degradations.gaussNoise('preprocessed.png')
            elif type == "Salt and Pepper Noise":
                controlled_degradations.saltpepper('preprocessed.png')
            elif type == "Blur":
                controlled_degradations.blur('preprocessed.png')
            elif type == "Reduce Illumination":
                controlled_degradations.debright('preprocessed.png')
            # open processed image
            if type == "None":
                pilimg = Image.open('preprocessed.png')
                self.nodegradechecker = True
            else:
                pilimg = Image.open('degraded.png')
            # resizing image to fit display
            pilimg.thumbnail((200, 200))
            gui_img = ImageTk.PhotoImage(pilimg)
            degraded = tk.Label(self, image=gui_img, text=f"{type} Degraded Image:", compound="top")
            degraded.grid(row=1, column=3, sticky="nsew", padx=10)
            degraded.image = gui_img
        else:
            tk.messagebox.showwarning(title="Warning",message="Please select a preprocessing step to begin.")
    ## function to create edge detected image from preprocessed image
    def edgedet(self, type):
        ### temp logging
        print("Edge Button Pushed")
        ###
        if os.path.exists("degraded.png") or self.nodegradechecker == True:
            if self.nodegradechecker == True:
                image = 'preprocessed.png'
            else:
                image = 'degraded.png'
            if type == "Prewitt":
                edge_detectors.prewitt(image)
            elif type == "Sobel":
                edge_detectors.sobel(image)
            elif type == "LoG":
                edge_detectors.log(image)
            elif type == "Canny":
                edge_detectors.canny(image)
            # open processed image
            if self.nodegradechecker == True:
                pilimg = Image.open('preprocessed.png')
            else:
                pilimg = Image.open('edge.png')
            # resizing image to fit display
            pilimg.thumbnail((200, 200))
            gui_img = ImageTk.PhotoImage(pilimg)
            edge = tk.Label(self, image=gui_img, text=f"{type} Edge Detected Image:", compound="top")
            edge.grid(row=1, column=4, sticky="nsew", padx=10)
            edge.image = gui_img
            #### HERE IS WHERE THE FINAL PART OF GUI IS BUILT TO COMPARE TO MASK

            ####
        else:
            tk.messagebox.showwarning(title="Warning", message="Please select a preprocessing step and degradation step (including none) to begin.")
    ## protocol function for destroying temporary images and gui
    def rmtemp(self):
        if os.path.exists('preprocessed.png'):
            os.remove('preprocessed.png')
        if os.path.exists('degraded.png'):
            os.remove('degraded.png')
        if os.path.exists('edge.png'):
            os.remove('edge.png')
        self.destroy()