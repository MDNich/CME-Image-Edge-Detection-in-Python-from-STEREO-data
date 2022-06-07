
from CMEContourHIimage import CMEContourHIimage

hiCME = CMEContourHIimage()

hiCME.loadImages("inputData/20081212_172901_2bh1B_br11.fts","inputData/20081212_180901_2bh1B_br11.fts","b")
#hiCME.loadImages("inputData/20081212_152901_2bh1B_br11.fts","inputData/20081212_160901_2bh1B_br11.fts","b")

hiCME.processCMEContour(True)

hiCME.drawCMEContour() # provide a string to the path where the image should 
                       # be saved if you want to save it and not just show.

# Comment this if you do not want to see intermediate steps
hiCME.drawProcDetail()
