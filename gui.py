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
import threading
import cv2


### Purpose of GUI: Example images for visualization, while showing entire results for whole dataset
## GUI Structure:
# row 1: available imgs scrollbar | original img  |   preprocessing  | degradation  | edge det
# row 2:         blank            |     blank     | preprocessed img | degraded img | edge det img
# row 3:     export options       | annotated img |     IoU img      |       testing results
class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
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

        # persistent references to re-usable control frames (built once in completeGUI)
        self._controls_built = False

        # custom close protocol
        self.protocol("WM_DELETE_WINDOW", self.rmtemp)

        # title management: track number of active processing calls so the
        # "Processing..." suffix is only removed when all of them have finished
        self._processing_count = 0
        self._base_title = "Robust Boundary Detection for Skin Lesions in Dermoscopic Images"
        self.title(self._base_title)

        # creating first GUI frame for image selector
        # start with canvas for scrollbar
        canvas = tk.Canvas(self, width=200)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="nsew")
        canvas.configure(yscrollcommand=scrollbar.set)

        # build frame with canvas
        selectorframe = tk.Frame(canvas, width=200, height=200)
        canvas.create_window((0, 0), window=selectorframe, anchor="nw")
        selectorframe.bind("<Configure>",
                           lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        selectorframe.grid(row=0, column=0, sticky="nsew", padx=10)
        selectorframe.grid_propagate(False)

        # recursively add buttons for every image in selector frame
        tk.Label(selectorframe, text="Available Images:").grid(row=0, column=0)
        imagedirectory = Path("image-loader")

        # create iteration variables:
        #   iteration        -> button row index
        #   buttoniteration  -> counts all dataset images to thin out displayed buttons
        iteration = 1
        buttoniteration = 1

        # list of image paths for evaluation of entire dataset
        self.dataset = []
        if not any(imagedirectory.iterdir()):
            tk.messagebox.showwarning(
                title="Warning",
                message="Please enter dataset into the 'image-loader' directory.")
            self.rmtemp()

        for images in imagedirectory.iterdir():
            imagename = images.name

            # filter out auxiliary ISIC dataset files
            if imagename.endswith('_superpixels.png'):
                continue
            elif imagename.endswith('_metadata.csv'):
                continue
            else:
                self.dataset.append(str(images))
                buttoniteration += 1

                # show only every 25th image as a button to avoid performance issues
                if buttoniteration % 25 == 0:
                    pilimage = Image.open(str(images))
                    pilimage = pilimage.resize((25, 25))
                    thumbnail = ImageTk.PhotoImage(pilimage)

                    item = tk.Button(selectorframe, image=thumbnail, text=imagename,
                                     compound="left",
                                     command=lambda imgpath=str(images): self.imageselect(imgpath))
                    item.grid(row=iteration, column=0, sticky="w")
                    item.image = thumbnail

                    selectorframe.grid(rowspan=iteration)
                    iteration += 1

                    canvas.create_window((0, iteration), window=selectorframe, anchor="nw")
                    selectorframe.configure(height=(33 * iteration))

    # -- title helpers --------------------------------------------------------

    def _begin_processing(self):
        """Increment the active-processing counter and update the window title."""
        self._processing_count += 1
        if self._processing_count == 1:
            self.title(self._base_title + " — Processing, Please Wait…")

    def _end_processing(self):
        """Decrement the active-processing counter and restore the title when idle."""
        self._processing_count = max(0, self._processing_count - 1)
        if self._processing_count == 0:
            self.title(self._base_title)

    # -- image selection ------------------------------------------------------

    def imageselect(self, imgpath):
        """Select the image to process and refresh the display."""
        self.currentimage = imgpath
        originalimg = Image.open(imgpath)
        originalimg.thumbnail((200, 200))
        gui_img = ImageTk.PhotoImage(originalimg)
        origlabel = tk.Label(self, image=gui_img, text="Original Image:", compound="top")
        origlabel.grid(row=0, column=2, sticky="nsew")
        origlabel.image = gui_img

        # build control panels on first use; they persist for the session
        if not self._controls_built:
            self.completeGUI()
            self._controls_built = True

        # propagate changes if a pipeline is already active
        if self.preprocesscheck:
            self.preprocess(self.lastpreprocess)

    # -- control panel construction (called once) -----------------------------

    def completeGUI(self):
        """Build the preprocessing, degradation, and edge-detection control panels.
        Called once after the first image is selected; subsequent image selections
        reuse the same widgets."""

        ## preprocessor
        preprocessorframe = tk.Frame(self, borderwidth=5, relief="groove",
                                     width=200, height=200)
        preprocessorframe.grid(row=0, column=3, sticky="nsew", padx=10)
        tk.Label(preprocessorframe, text="Preprocessing Tools:").grid(row=0, column=0)
        tk.Button(preprocessorframe, text="Gaussian Smoothing",
                  command=lambda: self.preprocess("Gaussian Smoothing")).grid(
                      row=1, column=0, sticky="nsew")
        tk.Button(preprocessorframe, text="Median Filtering",
                  command=lambda: self.preprocess("Median Filtering")).grid(
                      row=2, column=0, sticky="nsew")
        tk.Button(preprocessorframe, text="Bilateral Filtering",
                  command=lambda: self.preprocess("Bilateral Filtering")).grid(
                      row=3, column=0, sticky="nsew")
        tk.Button(preprocessorframe, text="Histogram Equalization",
                  command=lambda: self.preprocess("Histogram Equalization")).grid(
                      row=4, column=0, sticky="nsew")
        tk.Button(preprocessorframe,
                  text="Contrast Limited Adaptive Histogram Equalization",
                  command=lambda: self.preprocess("CLAHE")).grid(
                      row=5, column=0, sticky="nsew")
        tk.Button(preprocessorframe, text="None",
                  command=lambda: self.preprocess("None")).grid(
                      row=6, column=0, sticky="nsew")

        ## controlled degradations
        degradationframe = tk.Frame(self, borderwidth=5, relief="groove",
                                    width=200, height=200)
        degradationframe.grid(row=0, column=4, sticky="nsew", padx=10)
        tk.Label(degradationframe, text="Controlled Degradation Tools:").grid(row=0, column=0)
        tk.Button(degradationframe, text="None",
                  command=lambda: self.degrade("None")).grid(row=1, column=0, sticky="nsew")
        tk.Button(degradationframe, text="Gaussian Noise",
                  command=lambda: self.degrade("Gaussian Noise")).grid(
                      row=2, column=0, sticky="nsew")
        tk.Button(degradationframe, text="Salt and Pepper Noise",
                  command=lambda: self.degrade("Salt and Pepper Noise")).grid(
                      row=3, column=0, sticky="nsew")
        tk.Button(degradationframe, text="Blur",
                  command=lambda: self.degrade("Blur")).grid(row=4, column=0, sticky="nsew")
        tk.Button(degradationframe, text="Reduce Illumination",
                  command=lambda: self.degrade("Reduce Illumination")).grid(
                      row=5, column=0, sticky="nsew")

        ## edge detectors
        edgeframe = tk.Frame(self, borderwidth=5, relief="groove",
                             width=200, height=200)
        edgeframe.grid(row=0, column=5, sticky="nsew", padx=10)
        tk.Label(edgeframe, text="Edge Detection Tools:").grid(row=0, column=0)
        tk.Button(edgeframe, text="Sobel",
                  command=lambda: self.edgedet("Sobel")).grid(row=1, column=0, sticky="nsew")
        tk.Button(edgeframe, text="Prewitt",
                  command=lambda: self.edgedet("Prewitt")).grid(row=2, column=0, sticky="nsew")
        tk.Button(edgeframe, text="Laplacian of Gaussian",
                  command=lambda: self.edgedet("LoG")).grid(row=3, column=0, sticky="nsew")
        tk.Button(edgeframe, text="Canny Edge Detection",
                  command=lambda: self.edgedet("Canny")).grid(row=4, column=0, sticky="nsew")

    # -- pipeline stages ------------------------------------------------------

    def preprocess(self, type):
        """Apply the selected preprocessor to the current image."""
        self._begin_processing()
        self.nopreprocesschecker = False

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

        if type == "None":
            pilimg = Image.open(self.currentimage)
            self.nopreprocesschecker = True
        else:
            pilimg = Image.open('preprocessed.png')
        self.preprocessedimage = np.asarray(pilimg.convert("RGB"))

        pilimg.thumbnail((200, 200))
        gui_img = ImageTk.PhotoImage(pilimg)
        preprocesslabel = tk.Label(self, image=gui_img,
                                   text=f"{type} Preprocessed Image:", compound="top")
        preprocesslabel.grid(row=1, column=3, sticky="nsew", padx=10)
        preprocesslabel.image = gui_img

        self.preprocesscheck = True
        self.lastpreprocess = type
        if self.degradecheck:
            self.degrade(self.lastdegradation)
        self._end_processing()

    def degrade(self, type):
        """Apply the selected degradation to the preprocessed image."""
        self._begin_processing()
        self.nodegradechecker = False

        if os.path.exists("preprocessed.png") or self.nopreprocesschecker:
            image = self.currentimage if self.nopreprocesschecker else 'preprocessed.png'

            if type == "Gaussian Noise":
                controlled_degradations.gaussNoise(image)
            elif type == "Salt and Pepper Noise":
                controlled_degradations.saltpepper(image)
            elif type == "Blur":
                controlled_degradations.blur(image)
            elif type == "Reduce Illumination":
                controlled_degradations.debright(image)

            if type == "None":
                pilimg = Image.open(image)
                self.nodegradechecker = True
            else:
                pilimg = Image.open('degraded.png')
            self.degradedimage = np.asarray(pilimg.convert("RGB"))

            pilimg.thumbnail((200, 200))
            gui_img = ImageTk.PhotoImage(pilimg)
            degraded = tk.Label(self, image=gui_img,
                                text=f"{type} Degraded Image:", compound="top")
            degraded.grid(row=1, column=4, sticky="nsew", padx=10)
            degraded.image = gui_img

            self.degradecheck = True
            self.lastdegradation = type
            if self.edgedetected:
                self.edgedet(self.lastedge)
        else:
            tk.messagebox.showwarning(title="Warning",
                                      message="Please select a preprocessing step to begin.")
        self._end_processing()

    def edgedet(self, type):
        """Apply the selected edge detector to the (optionally degraded) image."""
        self._begin_processing()

        if os.path.exists("degraded.png") or self.nodegradechecker:
            if self.nodegradechecker:
                image = self.currentimage if self.nopreprocesschecker else 'preprocessed.png'
            else:
                image = 'degraded.png'

            if type == "Sobel":
                edge_detectors.sobel(image)
            elif type == "Prewitt":
                edge_detectors.prewitt(image)
            elif type == "LoG":
                edge_detectors.log(image)
            elif type == "Canny":
                edge_detectors.canny(image)

            pilimg = Image.open('edge.png')
            # Edge images are grayscale; convert to RGB so they can be stacked
            # with colour images during export without a shape mismatch
            self.edgedetectedimage = np.asarray(pilimg.convert("RGB"))

            pilimg.thumbnail((200, 200))
            gui_img = ImageTk.PhotoImage(pilimg)
            edge = tk.Label(self, image=gui_img,
                            text=f"{type} Edge Detected Image:", compound="top")
            edge.grid(row=1, column=5, sticky="nsew", padx=10)
            edge.image = gui_img

            self.edgedetected = True
            self.lastedge = type
            self.finalGUI()
        else:
            tk.messagebox.showwarning(
                title="Warning",
                message="Please select a preprocessing step and degradation step "
                        "(including none) to begin.")
        self._end_processing()

    # -- results panel --------------------------------------------------------

    def finalGUI(self):
        """Build (or refresh) the results row: annotated mask, IoU overlay,
        dataset statistics, and export buttons."""
        self._begin_processing()

        # Derive annotated mask path robustly via pathlib rather than a
        # hardcoded character-slice that assumes "image-loader/" is exactly
        # 13 characters long
        stem = Path(self.currentimage).stem
        annotatedpath = f"annotated-masks/{stem}_segmentation.png"
        self.annotatedimage = annotatedpath

        imagedirectory = Path("annotated-masks")
        if not any(imagedirectory.iterdir()):
            tk.messagebox.showwarning(
                title="Warning",
                message="Please enter ground-truth dataset into the 'annotated-masks' directory.")
            self.rmtemp()
        else:
            if not Path(annotatedpath).exists():
                tk.messagebox.showwarning(
                    title="Warning",
                    message=f"Annotated mask not found:\n{annotatedpath}")
                self._end_processing()
                return

            # display the contour-extracted boundary of the annotated mask
            # (matches what is actually used for scoring — not the raw filled region)
            anno_gray = cv2.imread(self.annotatedimage, cv2.IMREAD_GRAYSCALE)
            contour_arr = result._mask_to_contour_edge(anno_gray)
            # Scale 0/1 -> 0/255 and convert to RGB for display
            contour_disp = (contour_arr * 255).astype(np.uint8)
            annotatedimg = Image.fromarray(contour_disp, mode='L').convert('RGB')
            annotatedimg.thumbnail((200, 200))
            gui_img = ImageTk.PhotoImage(annotatedimg)
            annotatedlabel = tk.Label(self, image=gui_img,
                                      text="Annotated Mask (contour):", compound="top")
            annotatedlabel.grid(row=2, column=2, sticky="nsew")
            annotatedlabel.image = gui_img

            # compute and display IoU colour overlay
            # iou() now returns a dict: {iou, precision, recall, f1}
            metrics = result.iou("edge.png", self.annotatedimage, "disp")
            pilimg = Image.open('iou.png')
            pilimg.thumbnail((200, 200))
            gui_img2 = ImageTk.PhotoImage(pilimg)
            ioulabel = tk.Label(self, image=gui_img2,
                                text=(f"IoU: {metrics['iou']:.4f}  "
                                      f"P: {metrics['precision']:.4f}  "
                                      f"R: {metrics['recall']:.4f}  "
                                      f"F1: {metrics['f1']:.4f}"),
                                compound="top")
            ioulabel.grid(row=2, column=3, sticky="nsew")
            ioulabel.image = gui_img2

            # run full dataset evaluation in a background thread so the GUI
            # stays responsive; show a live progress label while it runs
            resultslabel = tk.Label(self,
                                    text="Running dataset evaluation...\n(scoring image 0 of "
                                         + str(len(self.dataset)) + ")",
                                    compound="top", justify="left", font=("Courier", 10))
            resultslabel.grid(row=2, column=4, sticky="nsew", columnspan=2)
            self._begin_processing()
            self._run_calculate_async(resultslabel)

            # export buttons (rebuilt each time to reflect the current pipeline)
            exportframe = tk.Frame(self, borderwidth=5, relief="groove",
                                   width=200, height=200)
            exportframe.grid(row=2, column=0, sticky="nsew", padx=10)
            tk.Label(exportframe, text="Export Options:").grid(row=0, column=0)
            tk.Button(exportframe, text="Export Final",
                      command=lambda: self.export("Final")).grid(
                          row=1, column=0, sticky="nsew")
            tk.Button(exportframe, text="Export Process",
                      command=lambda: self.export("Process")).grid(
                          row=2, column=0, sticky="nsew")
            tk.Button(exportframe, text="Export Final vs Annotated Mask",
                      command=lambda: self.export("FinalvAnno")).grid(
                          row=3, column=0, sticky="nsew")
            tk.Button(exportframe, text="Export Intersection over Union",
                      command=lambda: self.export("IoU")).grid(
                          row=4, column=0, sticky="nsew")
        self._end_processing()

    # -- background dataset evaluation ---------------------------------------

    def _run_calculate_async(self, label):
        """Run result.calculate() on a daemon thread and update *label* on completion.

        result.calculate() accepts a progress_callback(i, total) that is called
        after each image, allowing live label updates without a duplicate loop here.
        All processing is done in memory via the _arr pipeline — no temp files are
        touched during batch evaluation.
        """
        pre             = self.lastpreprocess
        deg             = self.lastdegradation
        edge            = self.lastedge
        total           = len(self.dataset)
        dataset_snapshot = list(self.dataset)

        def update_label(text):
            """Thread-safe label update via after()."""
            self.after(0, lambda t=text: label.config(text=t))

        def on_progress(i, total):
            update_label(
                f"Running dataset evaluation...\n"
                f"(scoring image {i} of {total})"
            )

        def worker():
            try:
                summary = result.calculate(
                    dataset_snapshot, pre, deg, edge,
                    progress_callback=on_progress
                )
                update_label(summary)
            except Exception as exc:
                update_label(f"Dataset evaluation failed:\n{exc}")
            finally:
                self.after(0, self._end_processing)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # -- export ---------------------------------------------------------------

    def export(self, type):
        """Save one of the four export types to a user-chosen file path."""
        # Derive a base filename from the original image stem via pathlib
        # (replaces the old hardcoded currentimage[13:] character slice)
        stem = Path(self.currentimage).stem

        if type == "Final":
            filename = f"{stem}_Final"
            final = Image.fromarray(self.edgedetectedimage)

        elif type == "Process":
            filename = f"{stem}_Process"

            # Collect whichever pipeline stages were active, all as RGB arrays
            main = np.asarray(Image.open(self.currentimage).convert("RGB"))
            third = self.edgedetectedimage          # already RGB (converted in edgedet)
            first = None if self.nopreprocesschecker  else self.preprocessedimage
            second = None if self.nodegradechecker    else self.degradedimage

            # Stack only the stages that were used; resize any that differ in height
            parts = [p for p in (main, first, second, third) if isinstance(p, np.ndarray)]
            h = parts[0].shape[0]
            resized = []
            for p in parts:
                if p.shape[0] != h:
                    pil_p = Image.fromarray(p).resize(
                        (int(p.shape[1] * h / p.shape[0]), h), Image.LANCZOS)
                    resized.append(np.asarray(pil_p))
                else:
                    resized.append(p)
            final = Image.fromarray(np.hstack(resized))

        elif type == "FinalvAnno":
            filename = f"{stem}_FinalvsAnnotated"
            first = self.edgedetectedimage           # RGB
            second = np.asarray(Image.open(self.annotatedimage).convert("RGB"))
            if first.shape[0] != second.shape[0]:
                h = first.shape[0]
                second = np.asarray(
                    Image.fromarray(second).resize(
                        (int(second.shape[1] * h / second.shape[0]), h), Image.LANCZOS))
            final = Image.fromarray(np.hstack((first, second)))

        elif type == "IoU":
            filename = f"{stem}_IoU"
            final = Image.open("iou.png")

        else:
            return

        file_path = tk.filedialog.asksaveasfilename(
            initialfile=f"{filename}.png",
            defaultextension=".png",
            filetypes=(("PNG files", "*.png"), ("All files", "*.*")),
            title="Export image to: "
        )
        if file_path:
            try:
                final.save(file_path)
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to save image: {e}")

    # -- cleanup --------------------------------------------------------------

    def rmtemp(self):
        """Remove temporary pipeline images and close the window."""
        for tmp in ('preprocessed.png', 'degraded.png', 'edge.png', 'iou.png'):
            if os.path.exists(tmp):
                os.remove(tmp)
        self.destroy()
