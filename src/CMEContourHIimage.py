"""
Module providing the CMEContourHIimage class to mark the edges of a CME image from
a difference of two consecutive STEREO-SECCHI Heliospheric Imager snapshots
Works best with 11-day background-removed HI1 images (left or right, that is
'a' or 'b' -- from the 'before' or 'after' spacecraft
"""

__all__ = ['CMEContourHIimage']

import sunpy.map

import matplotlib.colors as colors
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import patches

import scipy.ndimage as ndmg

# ==========================
        
class CMEContourHIimage():
        """
        A class to obtain the contour of CMEs as seen from two
        consecutive STEREO-SECCHI Heliospheric Imager snapshots
        """

        # book and house keeping: constructors, initializers, get, set parameters
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        def __init__(self,brightThresh=240,constrastThresh=120,angSteps=128,outlierSigmaThresh=2,windowSide=5,scaleFact=7,circMod=256,intersWin=20,denseWin=60,denseThreshFrac=0.5,runWin=5,runWinThresh=3,qRemoveOutliers=False,qTrace=True):
                self._brPxVal = brightThresh
                self._crPxVal = constrastThresh
                self._angSteps = angSteps
                self._outThrsh = outlierSigmaThresh
                self._qRemoveOutliers = qRemoveOutliers
                self._circSmWinS = 5
                self._circSmScaleF = 7
                self._circSmMod = circMod
                self._intersWin = intersWin
                self._denseWin = denseWin
                self._denseThreshFrac = 0.5
                self._halfWindMcheckBrightness = 3
                self._runWin = runWin
                self._runWinThresh = runWinThresh
                self._qTrace = qTrace
                self._spacecraft = None
                self._iniWinNcol = 8
                self._iniWinNrow = 3
                self._epsilon    = 0.000001
                self._chopMaxFrac = 0.15
                self._fracHdiam = 0.76
                self._factVdiam = 1.25


        def loadImages(self,file1,file2,spacecraft,qTrueCen=False,qDrawCenter=False):
                self._firstImageMap = sunpy.map.Map(file1)
                self._secondImageMap = sunpy.map.Map(file2)
                self._diffImageData  = 255*ndmg.uniform_filter(colors.LogNorm()(self._firstImageMap.data-self._secondImageMap.data).data,7)
                self._diffImageMap = sunpy.map.Map(self._diffImageData,self._secondImageMap.meta)
                self._spacecraft = spacecraft;
                if((self._spacecraft != 'a') and (self._spacecraft != 'b')):
                        print("ERROR: Spacecraft must be 'a' or 'b'")
                        exit(1)

                self._brCen = self.findBrightCenter(qTrueCen,qDrawCenter)
                #print("Diff image ready to explore from "+str(self._brCen))
                
        # to do: add get and set for each of these, and for any other such properties, params, etc
        # main processing 
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        def processCMEContour(self,qRemOut=False,qStop=False):

                self._beEdge = self.findBrightEndEdge(self._diffImageData,self._brCen,self._brPxVal)
                if(self._qTrace):
                        print(str(len(self._beEdge))+" bright edges")
                self._smoothImageData = self.circSmooth(self._diffImageData,self._circSmWinS,self._circSmScaleF,self._circSmMod)
                self._crEdge = self.findContrastEdge(self._smoothImageData,self._crPxVal)
                if(self._qTrace):
                        print(str(len(self._crEdge))+" contrast edges")

                self._inEdge,self._markedImageData = self.findEdgeSetIntersection(self._beEdge,self._crEdge,self._intersWin,np.shape(self._diffImageData))
                if(self._qTrace):
                        print(str(len(self._inEdge))+" interSet edges")

                self._dnEdge,self._denseImageData = self.findDenseEdges(self._markedImageData,self._denseWin,self._denseThreshFrac,np.shape(self._markedImageData))
                
                self._angInc   = 2*np.math.pi/self._angSteps
                self._outEdge,self._angleList,self._firstEdge = self.findIniContourFromCenterRad(self._denseImageData,self._brCen,self._angInc)
                if(self._qTrace):
                        print(str(len(self._outEdge))+" outer contour edges "+str(len(self._firstEdge))+" inner contour edges")

                self._chopEdge = []
                if(self._qRemoveOutliers or qRemOut):
                        self._newEdge,self._kIter,self._chopEdge = self.simpleRemoveOutliers(self._brCen,self._outEdge,self._angleList,self._firstEdge,self._outThrsh)

                        if(self._qTrace):
                                print(str(len(self._chopEdge))+" edges removed") 

                        if(len(self._chopEdge) > self._chopMaxFrac*len(self._outEdge)):
                                self._newEdge = self._outEdge
                                print("On second thought, restoring all removed edges")
                                
                else:
                        self._newEdge = self._outEdge;

                if(self.qCheckDiameters(self._newEdge,self._fracHdiam,self._factVdiam)):
                        print("The perimeter looks alright, approving")
                else:
                        print("The perimeter looks too sketchy, not trusting anything")
                        self._newEdge = []

                if(len(self._newEdge) == 0):
                        print("NOTHING WAS DETECTED")

                return

        # result output 
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        #: return the center and the edges as pixel coordinates
        #: return the center and the edges in world coordinates  ( <== to do )
        
        def drawCMEContour(self,outputImg="",qDontDraw=False):
                """
                draws a final-result picture with superimposed matplotlib artists
                """
                # to do: check for state (noImageLoaded,rawImageLoaded,imageProcessed,etc)
                ax=plt.subplot(projection=self._diffImageMap)
                im=self._diffImageMap.plot()
                if(len(self._newEdge)):
                        ax.add_artist(patches.Polygon(self._newEdge,color='blue',linewidth=2,fill=False,zorder=107))
                if(outputImg):
                        plt.savefig(outputImg+".pdf")

                if(not(qDontDraw)):
                        plt.show()
                
        def drawProcDetail(self):
                """
                draws details of internal processing stages with superimposed matplotlib artists
                """
                self._diffImageMap.plot()
                plt.show()
                ax=plt.subplot(projection=self._diffImageMap)
                im=self._diffImageMap.plot()
                for beE in self._beEdge:
                        ax.add_artist(patches.Circle(beE,7,color='red',zorder=102))
                
                for inE in self._inEdge:
                        ax.add_artist(patches.Circle(inE,7,color='orange',zorder=100))

                ax.add_artist(patches.Rectangle(self._brCen,9,9,color='black',zorder=104))

                plt.show()
                ax=plt.subplot(projection=self._diffImageMap)
                im=self._diffImageMap.plot()
                
                for dnE in self._dnEdge:
                        ax.add_artist(patches.Rectangle(dnE,1,1,color='orange',zorder=100))

                for outE in self._outEdge:
                        ax.add_artist(patches.Circle(outE,7,color='magenta',zorder=102))
                
                for chE in self._chopEdge:
                        ax.add_artist(patches.Rectangle(chE,5,5,color='black',zorder=104))

                ax.add_artist(patches.Rectangle(self._brCen,9,9,color='green',zorder=104))

                plt.show()

        # = = = = =

        # processing stages
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        def findBrightCenter(self,qTrueCen = False, qDraw = False):
                """
                finds the center of the zone with bright pixels, around the appropriate edge
                according to the spacecraft (a or b) and returns the point on its horizontal
                close to the edge (leaving room for the smoothing window)
                """
                imageRows,imageCols = np.shape(self._diffImageData)
                
                startCol = 0
                endCol   = int(imageCols/self._iniWinNcol)
                incCol   = 1
                nWinCol  = endCol-startCol+1
                startRow = int(imageRows/self._iniWinNrow)
                endRow   = (self._iniWinNrow-1)*startRow
                if(self._spacecraft == 'a'):
                        startCol = imageCols
                        endCol   = int((self._iniWinNcol-1)*imageCols/self._iniWinNcol)
                        incCol   = -1
                        nWinCol  = startCol-endCol

                cmMask = self._diffImageData > self._brPxVal
                
                cenRow,cenCol = ndmg.center_of_mass(self._diffImageData,cmMask)
                cenRow = int(cenRow)
                cenCol = int(cenCol)
                brCenF = (cenCol,cenRow)
                print("Center "+str(imageCols)+" "+str(imageRows)+" "+self._spacecraft+" w: "+str(startCol)+" "+str(endCol)+" : "+str(startRow)+" "+str(endRow)+" @ "+str(brCenF))
                if(qDraw):
                        ax=plt.subplot(projection=self._diffImageMap)
                        im=self._diffImageMap.plot()
                        ax.add_artist(patches.Polygon([[startCol,startRow],[endCol,startRow],[endCol,endRow],[startCol,endRow]],color='blue',linewidth=2,fill=False,zorder=107))
                        ax.add_artist(patches.Rectangle(brCenF,7,7,color='red',zorder=108))
                        plt.show()
                
                if(qTrueCen):
                        return(cenCol,cenRow)

                retCenCol = startCol+incCol*(int(self._intersWin/2)+1)
                while(((incCol > 0) and (retCenCol < cenCol)) or ((incCol < 0) and (retCenCol > cenCol))):
                        if(self._diffImageData[cenRow,retCenCol] > self._brPxVal):
                                return (retCenCol,cenRow)
                        retCenCol += incCol

                return (retCenCol,cenRow)

        
        def findBrightEndEdge(self,imageData,startCoord,edgeThreshold):
                """
                returns a list with col,row coordinates of the edges found
                """
                imageRows,imageCols = np.shape(imageData)
                startCol,startRow = startCoord
                firstCol = 0
                endCol = imageCols
                incCol = 1
                if(self._spacecraft == "a"):
                        firstCol = imageCols-1
                        endCol = -1
                        incCol = -1
                        
                edgeCoords = []
                print("Find: "+str(startCol)+" "+str(startRow)+" "+str(edgeThreshold)+" "+str(imageCols)+" "+str(imageRows))
                for kRow in range(0,imageRows):
                        edgeCol = startCol
                        qMoved = False
                        for kCol in range(startCol,endCol,incCol):
                                if(imageData[kRow,kCol] > edgeThreshold):
                                        edgeCol = kCol
                                        qMoved = True
                                
                        if(qMoved):
                                edgeCoords.append([edgeCol,kRow])
                                

                for kCol in range(firstCol,endCol,incCol):
                        edgeRow = startRow
                        qMoved = False
                        for kRow in range(startRow,imageRows):
                                if(imageData[kRow,kCol] > edgeThreshold):
                                        edgeRow = kRow
                                        qMoved = True

                        if(qMoved):
                                edgeCoords.append([kCol,edgeRow])

                        edgeRow = startRow
                        qMoved = False
                        for kRow in range(startRow,0,-1):
                                if(imageData[kRow,kCol] > edgeThreshold):
                                        edgeRow = kRow
                                        qMoved = True

                        if(qMoved):
                                edgeCoords.append([kCol,edgeRow])

                return edgeCoords

        def circSmoothWindow(self,imageData,mRow,mCol,halfWindM1,scaleFactor,circMod):
                """
                returns the smoothed pixel at the center of the window, obtained
                through averaging and modulo circMod
                """
                newValue = 0
                for kRow in range(mRow-halfWindM1,mRow+halfWindM1+1):
                        for kCol in range(mCol-halfWindM1,mCol+halfWindM1+1):
                                newValue += imageData[kRow,kCol]

                newValue *= scaleFactor
                return int(newValue)%circMod

        def circSmooth(self,imageData,windowSide,scaleFact,circMod):
                """
                returns the smoothed image through averaging and modulo circMod 
                """
                smoothImageData = np.array(imageData)
                halfWindM1 = int(windowSide/2)
                overallScale = scaleFact/(float(windowSide*windowSide))
                imageRows,imageCols = np.shape(imageData)
                for kRow in range(halfWindM1,imageRows-halfWindM1):
                        for kCol in range(halfWindM1,imageCols-halfWindM1):
                                smoothImageData[kRow,kCol] = self.circSmoothWindow(imageData,kRow,kCol,halfWindM1,overallScale,circMod)

                return smoothImageData

        def findContrastEdge(self,imageData,edgeThreshold):
                """
                returns a list with col,row coordinates of the edges found
                """
                edgeCoords = []
                imageRows,imageCols = np.shape(imageData)
                for kRow in range(imageRows):
                        for kCol in range(imageCols-1):
                                if(abs(imageData[kRow,kCol]-imageData[kRow,kCol+1])>edgeThreshold):
                                        edgeCoords.append([kCol,kRow])

                for kCol in range(imageCols):
                        for kRow in range(imageRows-1):
                                if(abs(imageData[kRow,kCol]-imageData[kRow+1,kCol])>edgeThreshold):
                                        edgeCoords.append([kCol,kRow])

                return edgeCoords

        def findEdgeSetIntersection(self,edgeC1,edgeC2,intersWin,imageShape):
                """
                returns a list of edges found where both edgeC1 and edgeC2 
                exist within a intersWin square as well as the image 
                data with the markers (values of at least 2), propagated 
                in a window around the initially found values
                """
                edgeCoords = []
                imageBuff1 = np.ones(imageShape)
                imageBuff2 = np.ones(imageShape)
                imageRows,imageCols = imageShape
                halfWindM2 = int(intersWin/2)
                for edg in edgeC1:
                        imageBuff1[edg[1],edg[0]] = 2

                for edg in edgeC2:
                        imageBuff2[edg[1],edg[0]] = 3

                imageBuffS = imageBuff1 + imageBuff2
                imageBuffI = ndmg.filters.generic_filter(imageBuffS,np.prod,intersWin)
                imageBuffJ = np.zeros(imageShape)
                for kRow in range(imageRows):
                        imageBuffJ[kRow,0] = 10
                        imageBuffJ[kRow,imageCols-1] = 10

                for kCol in range(imageCols):
                        imageBuffJ[0,kCol] = 10
                        imageBuffJ[imageRows-1,kCol] = 10
                        

                for kRow in range(imageRows):
                        kCol = 0
                        while(kCol < imageCols):
                                if(((imageBuffI[kRow,kCol]%2 == 0) and (imageBuffI[kRow,kCol]%3 == 0)) or (imageBuffI[kRow,kCol]%5 == 0)):
                                        imageBuffJ[kRow,kCol] += 1
                                        kCol += halfWindM2 
                                else:
                                        kCol += 1

                for kCol in range(imageCols):
                        kRow = 0
                        while(kRow < imageRows):
                                if(((imageBuffI[kRow,kCol]%2 == 0) and (imageBuffI[kRow,kCol]%3 == 0))  or (imageBuffI[kRow,kCol]%5 == 0)):
                                        imageBuffJ[kRow,kCol] += 1
                                        kRow += halfWindM2
                                else:
                                        kRow += 1

                for kRow in range(0,imageRows): 
                        for kCol in range(0,imageCols): 
                                if(imageBuffJ[kRow,kCol] > 1):
                                        edgeCoords.append([kCol,kRow])        

                imageBuffK = ndmg.maximum_filter(imageBuffJ,size=halfWindM2)
                return edgeCoords,imageBuffK


        def findDenseEdges(self,imageData,denseWin,denseThreshFrac,imageShape):
                imageRows,imageCols = imageShape
                imageBuffA = (imageData > 1)*1
                imageBuffB = ndmg.filters.generic_filter(imageBuffA,np.sum,denseWin)
                denseThresh = denseThreshFrac*denseWin**2
                imageBuffC = (imageBuffB > denseThresh)*2
                edgeCoords = []
                for kRow in range(0,imageRows): 
                        for kCol in range(0,imageCols): 
                                if(imageBuffC[kRow,kCol] > 1):
                                        edgeCoords.append([kCol,kRow])  

                return edgeCoords,imageBuffC
        
        def findIniContourFromCenterRad(self,imageData,cenPx,angInc):
                """
                Returns the list of intersection edges, as marked with imageData, 
                radially found around the given center
                """
                retLists = [[],[],[]]
                for angle in np.arange(0,2*np.math.pi,angInc):
                        lastCol,lastRow,angle,firstCol,firstRow = self.goAlongOneRadius(imageData,cenPx,angle,None,False,self.capRadWithLastOne,None)
                        if(lastCol != None):
                                retLists[0].append([lastCol,lastRow])
                                retLists[1].append(angle)
                                retLists[2].append([firstCol,firstRow])

                return retLists
        
        def capRadWithLastOne(self,imageData,kCol,kRow,angle,dmrg):
                return (imageData[kRow,kCol] > 1)

        def checkBrightnessAlong(self,imageData,kCol,kRow,angle,brightThreshold):
                """
                checks whether in a small window we find bright pixels, and returns
                True and the center coordinates if we are below the brightness threshold
                This function is meant to be called as the 'iterFun' for radSwingAroundFromCenter()
                (see further)
                """
                halfWindMcheck = self._halfWindMcheckBrightness 
                nPxWind = float((2*halfWindMcheck + 1)**2)
                brightValue = 0
                for qRow in range(kRow-halfWindMcheck,kRow+halfWindMcheck+1):
                        for qCol in range(kCol-halfWindMcheck,kCol+halfWindMcheck+1):
                                brightValue += imageData[qRow,qCol]

                brightValue = float(brightValue) / nPxWind 
                if(brightValue < brightThreshold):
                        return True
                return False

        def qEdgeTooFarOut(self,imageData,cenPx,angle,radius):
                """
                Brings a far out edge closer by stopping when the brightness diminishes
                when walking outward along a radius from the center 
                """
                brTh = self._brPxVal
                sCol,sRow,angle = self.goAlongOneRadius(imageData,cenPx,angle,radius,True,self.checkBrightnessAlong,self._brPxVal)
                if(sCol == None):
                        return False,0,0
                
                return True,sCol,sRow

        def qStillShorterThan(self,maxRadius,cenPx,kCol,kRow):
                if(maxRadius == None):
                        return True
                cenCol,cenRow = cenPx
                if(np.sqrt((kCol-cenCol)**2+(kRow-cenRow)**2) < maxRadius):
                        return True
                return False

        def goAlongOneRadius(self,imageData,cenPx,angle,maxRadius,qFirst,iterFun,iterFunArg):
                """
                Explores one radial line for the given angle, from the center outward
                returning the col,row,angle of the first or last good point as deemed by iterFun
                """

                rowInc = 0
                colInc = 0
                cenCol,cenRow = cenPx
                imageRows,imageCols = np.shape(imageData)
                qVert  = False
                if(np.math.fabs(angle-np.math.pi/2) < self._epsilon):
                        rowInc = 1
                        qVert = True
                else:
                        if(np.math.fabs(angle-3*np.math.pi/2) < self._epsilon):
                                rowInc = -1
                                qVert = True
                        else:
                                if((angle < np.math.pi/2) or (angle > 3*np.math.pi/2)):
                                        if((angle <= np.math.pi/4) or (angle >= 7*np.math.pi/4)):
                                                colInc = 1
                                        else:
                                                if(angle < np.math.pi/2):
                                                        rowInc = 1
                                                else:
                                                        rowInc = -1
                                else:
                                        if((angle >= 3*np.math.pi/4) or (angle <= 5*np.math.pi/4)):
                                                colInc = -1
                                        else:
                                                if(angle < 3*np.math.pi/4):
                                                        rowInc = 1
                                                else:
                                                        rowInc = -1

                if((rowInc == 0) and (colInc == 0)):
                        print("ERROR Internal inconsistency: no increment")
                        exit(1)
                if((rowInc != 0) and (colInc != 0)):
                        print("ERROR Internal inconsistency: two increments")
                        exit(2)
                kRow = cenRow
                kCol = cenCol
                lastColRet = None
                lastRowRet = None
                firstColRet = None
                firstRowRet = None
                midColRet = None
                midRowRet = None
                qNothingYet = True
                qNoMidYet = True
                #retVal = None,None,None
                while((kRow < imageRows) and (kRow >= 0) and (kCol < imageCols) and (kCol >= 0) and self.qStillShorterThan(maxRadius,cenPx,kCol,kRow)):
                        if(colInc != 0):
                                kRow = int(np.math.tan(angle) * (kCol - cenCol) + cenRow)
                                if((kRow >= imageRows) or (kRow < 0)):
                                        break
                        else:
                                if(not(qVert)):
                                        kCol = int(1/np.math.tan(angle) * (kRow - cenRow) + cenCol)
                                        if((kCol >= imageCols) or (kCol < 0)):
                                                break

                        #qBreak,accumVal = iterFun(imageData,kCol,kRow,angle,accumVal)
                        
                        qGood = iterFun(imageData,kCol,kRow,angle,iterFunArg)
                        if(qGood):
                                lastColRet = kCol
                                lastRowRet = kRow
                                #retVal = kCol,kRow,angle

                                if(qNothingYet):
                                        firstColRet = kCol
                                        firstRowRet = kRow
                                        qNothingYet = False
                                
                        if(qFirst):
                                break

                        if(not(qNothingYet) and not(qGood) and qNoMidYet):
                                midColRet = kCol
                                midRowRed = kRow
                                qNoMidYet = False
                        
                        kRow += rowInc
                        kCol += colInc

 
                return lastColRet,lastRowRet,angle,firstColRet,firstRowRet#,midColRet,midRowRet



        def edgeRadVals(self,cenPx,edgeCoords):
                """
                returns the array of radii for each edge
                """
                cenX,cenY = cenPx
                nEdges = len(edgeCoords)
                radVals = np.zeros(nEdges)
                for vX in range(nEdges):
                        radVals[vX] = np.sqrt((edgeCoords[vX][0]-cenX)**2 + (edgeCoords[vX][1]-cenY)**2)

                return radVals
        

        def avgCircleRadStats(self,cenPx,radVals,outThreshold):
                """
                computes radius stats with respect to average circle 
                """
                avgRad = radVals.mean()
                avgStd = radVals.std()
                outCoords = np.zeros(len(radVals),dtype=np.int8)
                cenX,cenY = cenPx
                threshStd = outThreshold*avgStd
                for vX in range(len(radVals)):
                        if(abs(radVals[vX]-avgRad) > threshStd):
                                outCoords[vX] = 1

                return avgRad,avgStd,outCoords


        def runningAvgCircleRadStats(self,radVals,runWin,runThreshold):
                """
                Marks outliers in a circularly running window 
                """
                nIniVals = len(radVals)
                halfRunWin = int((runWin-1)/2)
                nCircVals = nIniVals+2*halfRunWin
                circRadVals = np.zeros(nCircVals)
                outCoords = np.zeros(nIniVals)
                for vX in range(halfRunWin):
                        circRadVals[vX] = radVals[nIniVals-halfRunWin+vX]
                for vX in range(nIniVals):
                        circRadVals[halfRunWin+vX] = radVals[vX]
                for vX in range(halfRunWin):
                        circRadVals[nIniVals+halfRunWin+vX] = radVals[vX]

                for vX in range(nIniVals):
                        avgAcc = float(0.0)
                        avgAcc3 = float(0.0)
                        for aX in range(runWin):
                                if(aX != halfRunWin):
                                        avgAcc += circRadVals[vX+aX]

                                if((aX != halfRunWin-1) and (aX != halfRunWin) and (aX != halfRunWin+1)):
                                        avgAcc3 += circRadVals[vX+aX]
                        avgAcc = avgAcc/float(runWin-1)
                        avgAcc3 = avgAcc3/float(runWin-3)
                        stdAcc = float(0.0)
                        stdAcc3 = float(0.0)
                        for aX in range(runWin):
                                if(aX != halfRunWin):
                                        stdAcc += (avgAcc-circRadVals[vX+aX])**2
                                if((aX != halfRunWin-1) and (aX != halfRunWin) and (aX != halfRunWin+1)):
                                        stdAcc3 += (avgAcc3-circRadVals[vX+aX])**2
                        stdAcc = np.sqrt(stdAcc/(runWin-1))
                        stdAcc3 = np.sqrt(stdAcc3/(runWin-3))
                        if(abs(circRadVals[vX+halfRunWin]-avgAcc) > runThreshold*stdAcc):
#                           or (abs(circRadVals[vX+halfRunWin]-avgAcc3) > runThreshold*stdAcc3)):
                                outCoords[vX] = 1

                return outCoords
        
        def simpleRemoveOutliers(self,brCen,outEdge,angleList,firstEdge,outThrsh):
                """
                Removes major outliers from the given contour
                """
                outRadVals = self.edgeRadVals(brCen,outEdge)
                firstRadVals = self.edgeRadVals(brCen,firstEdge)

                iniAvgRad,iniAvgStd,outCoords = self.avgCircleRadStats(brCen,outRadVals,outThrsh)
                diffAvgRad,diffAvgStd,diffCoords = self.avgCircleRadStats(brCen,outRadVals-firstRadVals,outThrsh)
                
                print("Found "+str(outCoords.sum())+" out outliers "+str(diffCoords.sum())+" diff outliers")
                
                chopOutEdge = []
                newEdge = []
                if(len(outEdge) != len(outCoords)):
                        print("ERROR Inconsistent lengths")
                        exit(1)

                if(len(outEdge) != len(diffCoords)):
                        print("ERROR Inconsistent diff lengths")
                        exit(2)

                
                for kEdge in range(0,len(outEdge)):
                        if(outCoords[kEdge] and diffCoords[kEdge]):
                                chopOutEdge.append(outEdge[kEdge])
                        else:
                                newEdge.append(outEdge[kEdge])

                ok1RadVals = self.edgeRadVals(brCen,newEdge)
                circCoords = self.runningAvgCircleRadStats(ok1RadVals,self._runWin,self._runWinThresh)
                print("Found "+str(circCoords.sum())+" circ outliers")
                if(len(newEdge) != len(circCoords)):
                        print("ERROR Inconsistent circ lengths")
                        exit(3)

                goodEdge = []
                        
                for kEdge in range(0,len(newEdge)):
                        if(circCoords[kEdge]):
                                chopOutEdge.append(newEdge[kEdge])
                        else:
                                goodEdge.append(newEdge[kEdge])

                return goodEdge,1,chopOutEdge


        def qCheckDiameters(self,edge,fracHdiam,factVdiam):
                if(len(edge)<3):
                        return False

                qLooksOk = True
                nEdges = len(edge)
                xE = np.zeros(nEdges)
                yE = np.zeros(nEdges)
                for kE in range(nEdges):
                        xE[kE] = edge[kE][0]
                        yE[kE] = edge[kE][1]

                maxXe = xE.max()
                minXe = xE.min()
                xDiam = maxXe-minXe

                xARestr = int(fracHdiam*xDiam)
                xBRestr = int((1-fracHdiam)*xDiam)
                
                if(self._spacecraft == "a"):
                        firstX = xE > xARestr

                else:
                        firstX = xE < xBRestr

                yRestr = yE[firstX]
                minYe = yRestr.min()
                maxYe = yRestr.max()
                yDiam = maxYe - minYe
                thrsh4x = int(factVdiam*yDiam)
                print("Checking horiz "+str(xDiam)+" vert "+str(yDiam)+" "+str(thrsh4x))
                if(xDiam > thrsh4x):
                        print("Too long to be true")
                        qLooksOk = False
                else:
                        print("Looks good")

                return qLooksOk
                
