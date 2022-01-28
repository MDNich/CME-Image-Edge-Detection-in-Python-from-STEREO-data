
from CMEContourHIimage import CMEContourHIimage

hiCME = CMEContourHIimage()

hiCME.loadImages("inputData/20081212_172901_2bh1B_br11.fts","inputData/20081212_180901_2bh1B_br11.fts","b")

hiCME.processCMEContour(True)

hiCME.drawCMEContour() # provide a string to the path where the image should 
                       # be saved if you want to save it and not just show.


