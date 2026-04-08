import tkinter as tk
import tkinter.messagebox
import tkinter.filedialog
from pathlib import Path
import numpy as np
from PIL import Image, ImageTk
import pre_processors
import controlled_degradations
import edge_detectors
import result
import os


##### Remaining for this file: removing logs, update naming conventions for annotated masks
##### need to find dataset with annotated masks or way to convert ISIC or kaggle dataset

### Purpose of GUI: Example images for visualization, while showing entire results for whole dataset
## GUI Structure:
# row 1: available imgs scrollbar | original img  |   preprocessing  | degradation  | edge det
# row 2:         blank            |     blank     | preprocessed img | degraded img | edge det img
# row 3:     export options       | annotated img |     IoU img      |       testing results
class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        # RBD-SLDI
        self.title("Robust Boundary Detection for Skin Lesions in Dermoscopic Images")
        self.minsize(1200, 800)
        # placeholder values to carry the imgpath of the currently targeted image
        self.currentimage = None
        self.annotatedimage = None
        self.preprocessedimage = None
        self.degradedimage = None
        self.edgedetectedimage = None
        # checker values for none type preprocess/degrade
        self.nodegradechecker = False
        self.nopreprocesschecker = False
        # checker values for processing, used for propagating changes downwards
        self.preprocesscheck = False
        self.degradecheck = False
        self.edgedetected = False
        # placeholder values to carry types for propagating changes
        self.lastpreprocess = None
        self.lastdegradation = None
        self.lastedge = None
        # custom close protocol
        self.protocol("WM_DELETE_WINDOW", self.rmtemp)
        # creating first GUI frame for image selector
        # start with canvas for scrollbar
        canvas = tk.Canvas(self, width=200)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="nsew")
        canvas.configure(yscrollcommand=scrollbar.set)
        # build frame with canvas
        selectorframe = tk.Frame(canvas, width=200, height=200)
        canvas.create_window((0,0), window=selectorframe, anchor="nw")
        selectorframe.bind("<Configure>",lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        selectorframe.grid(row=0, column=0, sticky="nsew", padx=10)
        selectorframe.grid_propagate(False)
        # recursively add buttons for every image in selector frame
        tk.Label(selectorframe,text="Available Images:").grid(row=0, column=0)
        imagedirectory = Path("image-loader")
        iteration = 1
        # list of imagepaths for evaluations of entire dataset
        self.dataset = []
        for images in imagedirectory.iterdir():
            # button label name
            imagename = images.name
            # add to dataset
            self.dataset.append(str(images))
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
            # use larger iteration to update on canvas and reconfigure scroll distance
            canvas.create_window((0,iteration), window=selectorframe, anchor="nw")
            selectorframe.configure(height=(32*iteration))
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
        origlabel.grid(row=0, column=2, sticky="nsew")
        origlabel.image = gui_img
        # build the rest of the GUI frames and buttons
        self.completeGUI()
        # propagate changes if the program is already in use
        if self.preprocesscheck == True:
            self.preprocess(self.lastpreprocess)
    ## function for building the rest of the GUI after an original image has been selected
    def completeGUI(self):
        ### additional frames for each processing type and their corresponding buttons
        ## preprocessor
        preprocessorframe = (tk.Frame(self, borderwidth=5, relief="groove",
                                     width=200, height=200))
        preprocessorframe.grid(row=0, column=3, sticky="nsew", padx=10)
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
        clahe.grid(row=5, column=0, sticky="nsew")
        # None
        none = tk.Button(preprocessorframe, text="None",
                         command=lambda: self.preprocess("None"))
        none.grid(row=6, column=0, sticky="nsew")
        ## controlled degradations
        degradationframe = tk.Frame(self, borderwidth=5, relief="groove",
                                    width=200, height=200)
        degradationframe.grid(row=0, column=4, sticky="nsew", padx=10)
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
                             width=200, height=200)
        edgeframe.grid(row=0, column=5, sticky="nsew", padx=10)
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
        self.nopreprocesschecker = False
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
        if type == "None":
            pilimg = Image.open(self.currentimage)
            self.nopreprocesschecker = True
        else:
            pilimg = Image.open('preprocessed.png')
        self.preprocessedimage = np.asarray(pilimg)
        # resizing image to fit display
        pilimg.thumbnail((200, 200))
        gui_img = ImageTk.PhotoImage(pilimg)
        preprocesslabel = tk.Label(self, image=gui_img, text=f"{type} Preprocessed Image:", compound="top")
        preprocesslabel.grid(row=1, column=3, sticky="nsew", padx=10)
        preprocesslabel.image = gui_img
        # update leaf images
        self.preprocesscheck = True
        self.lastpreprocess = type
        if self.degradecheck == True:
            self.degrade(self.lastdegradation)
    ## function to create degraded images from preprocessed image
    def degrade(self, type):
        ### temp logging
        print("Degrade Button Pushed")
        ###
        self.nodegradechecker = False
        if os.path.exists("preprocessed.png") or self.nopreprocesschecker == True:
            if self.nopreprocesschecker == True:
                image = self.currentimage
            else:
                image = 'preprocessed.png'
            if type == "Gaussian Noise":
                controlled_degradations.gaussNoise(image)
            elif type == "Salt and Pepper Noise":
                controlled_degradations.saltpepper(image)
            elif type == "Blur":
                controlled_degradations.blur(image)
            elif type == "Reduce Illumination":
                controlled_degradations.debright(image)
            # open processed image
            if type == "None":
                pilimg = Image.open(image)
                self.nodegradechecker = True
            else:
                pilimg = Image.open('degraded.png')
            self.degradedimage = np.asarray(pilimg)
            # resizing image to fit display
            pilimg.thumbnail((200, 200))
            gui_img = ImageTk.PhotoImage(pilimg)
            degraded = tk.Label(self, image=gui_img, text=f"{type} Degraded Image:", compound="top")
            degraded.grid(row=1, column=4, sticky="nsew", padx=10)
            degraded.image = gui_img
            # update leaf images
            self.degradecheck = True
            self.lastdegradation = type
            if self.edgedetected == True:
                self.edgedet(self.lastedge)
        else:
            tk.messagebox.showwarning(title="Warning",message="Please select a preprocessing step to begin.")
    ## function to create edge detected image from preprocessed image
    def edgedet(self, type):
        ### temp logging
        print("Edge Button Pushed")
        ###
        self.edgedetected = True
        if os.path.exists("degraded.png") or self.nodegradechecker == True:
            if self.nodegradechecker == True:
                if self.nopreprocesschecker == True:
                    image = self.currentimage
                else:
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
            pilimg = Image.open('edge.png')
            self.edgedetectedimage = np.asarray(pilimg)
            # resizing image to fit display
            pilimg.thumbnail((200, 200))
            gui_img = ImageTk.PhotoImage(pilimg)
            edge = tk.Label(self, image=gui_img, text=f"{type} Edge Detected Image:", compound="top")
            edge.grid(row=1, column=5, sticky="nsew", padx=10)
            edge.image = gui_img
            # create type for previous branch changes
            self.edgedetected = True
            self.lastedge = type
            # Build final stage of GUI
            self.finalGUI()
        else:
            tk.messagebox.showwarning(title="Warning", message="Please select a preprocessing step and degradation step (including none) to begin.")
    ## complete GUI
    def finalGUI(self):
        # find annotated mask and display
        ###### UPDATE NAMING CONVENTIONS ######
        # direct original image path to annotated-masks folder for annotated mask
        annotatedpath = "annotated-masks/"
        annotatedpath+=self.currentimage[13:]
        self.annotatedimage = annotatedpath
        #### temp logging
        print(self.annotatedimage)
        ####
        # display annotated mask on GUI
        annotatedimg = Image.open(self.annotatedimage)
        # resizing image to fit display
        annotatedimg.thumbnail((200, 200))
        gui_img = ImageTk.PhotoImage(annotatedimg)
        annotatedlabel = tk.Label(self, image=gui_img, text="Annotated Mask:", compound="top")
        annotatedlabel.grid(row=2, column=2, sticky="nsew")
        annotatedlabel.image = gui_img
        # display iou mask with final image
        # manually call result function with display type
        result.iou("edge.png", self.annotatedimage, "disp")
        # open new iou image and display
        pilimg = Image.open('iou.png')
        pilimg.thumbnail((200, 200))
        gui_img2 = ImageTk.PhotoImage(pilimg)
        ioulabel = tk.Label(self, image = gui_img2, text="Intersection over Union:", compound="top")
        ioulabel.grid(row=2, column=3, sticky="nsew")
        ioulabel.image = gui_img2
        # obtain dataset results
        results = result.calculate(self.dataset, self.lastpreprocess, self.lastdegradation, self.lastedge)
        resultslabel = tk.Label(self, text=results, compound="top")
        resultslabel.grid(row=2, column=4, sticky="nsew", columnspan=2)
        ## export buttons
        exportframe = (tk.Frame(self, borderwidth=5, relief="groove",
                                      width=200, height=200))
        exportframe.grid(row=2, column=0, sticky="nsew", padx=10)
        tk.Label(exportframe, text="Export Options:").grid(row=0, column=0)
        export1 = tk.Button(exportframe, text="Export Final",
                          command=lambda: self.export("Final"))
        export1.grid(row=1, column=0, sticky="nsew")
        export2 = tk.Button(exportframe, text="Export Process",
                            command=lambda: self.export("Process"))
        export2.grid(row=2, column=0, sticky="nsew")
        export3 = tk.Button(exportframe, text="Export Final vs Annotated Mask",
                            command=lambda: self.export("FinalvAnno"))
        export3.grid(row=3, column=0, sticky="nsew")
        export4 = tk.Button(exportframe, text="Export Intersection over Union",
                            command=lambda: self.export("IoU"))
        export4.grid(row=4, column=0, sticky="nsew")
    ## export function
    def export(self, type):
        print("Export Button Pushed")
        # take original image name to concatenate with export strings
        newimg = self.currentimage[13:]
        newimg = Path(newimg).stem
        if type == "Final":
            # change export string
            newimg+="_Final"
            print(f"Saving as: {newimg}")
            # pull image from stored array
            final = Image.fromarray(self.edgedetectedimage)
        elif type == "Process":
            # change export string
            newimg += "_Process"
            print(f"Saving as: {newimg}")
            # open original image with 3 color channels
            main = Image.open(self.currentimage)
            main = main.convert("RGB")
            main = np.asarray(main)
            # open edge detected image array, mandatory step so no logic required
            third = self.edgedetectedimage
            # open preprocessed and degraded image arrays, if their selection are not none
            first = None
            second = None
            if self.nopreprocesschecker != True:
                first = self.preprocessedimage
            if self.nodegradechecker != True:
                second = self.degradedimage
            # concatenate images side by side according to which options were used
            # if a preprocessor or degradation was not selected, it is not included
            if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
                exportablearray = np.hstack((main,first,second,third))
            elif isinstance(first, np.ndarray) and not isinstance(second, np.ndarray):
                exportablearray = np.hstack((main,first,third))
            elif not isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
                exportablearray = np.hstack((main,second,third))
            else:
                exportablearray = np.hstack((main,third))
            # convert to image to be saved
            final = Image.fromarray(exportablearray)
        elif type == "FinalvAnno":
            newimg += "_FinalvsAnnotated"
            print(f"Saving as: {newimg}")
            # pull image from stored array
            first = self.edgedetectedimage
            # pull image from stored image path
            second = Image.open(self.annotatedimage)
            second = second.convert("RGB")
            second = np.asarray(second)
            # combine images horizontally and convert to saveable image
            exportablearray = np.hstack((first, second))
            final = Image.fromarray(exportablearray)
        elif type == "IoU":
            newimg += "_IoU"
            print(f"Saving as: {newimg}")
            # open iou image (Created from final gui creation steps)
            final = Image.open("iou.png")
        # save as popup
        file_path = tk.filedialog.asksaveasfilename(
            initialfile=f"{newimg}.png",
            defaultextension=".png",
            filetypes=(("PNG files", "*.png"), ("All files", "*.*")),
            title="Export image to: "
        )
        if file_path:
            try:
                final.save(file_path)
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to save image: {e}")
    ## protocol function for destroying temporary images and gui
    def rmtemp(self):
        if os.path.exists('preprocessed.png'):
            os.remove('preprocessed.png')
        if os.path.exists('degraded.png'):
            os.remove('degraded.png')
        if os.path.exists('edge.png'):
            os.remove('edge.png')
        if os.path.exists('iou.png'):
            os.remove('iou.png')
        self.destroy()