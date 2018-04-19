# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:29:46 2017

@author: Kia Akrami
File used to process the new CSV files output by XRFd.
For use by field personnel to review SX1 data.

Revision 1.35 - Changes to work with new logs
Revision 1.34 - Updated kilocounts plot to avoid misleading y-axis in the left plot of counts and laser subplot
                Added input variavle "numChannel" that has the number of channels the project has
Revision 1.33 - Updated to discard first 1.5s and last 0.5s of recordings when counting bad samples
Revision 1.32 - Updated to fix problem with 100x on dead time
Revision 1.31 - Updated to fix problem with file naming which caused file corruption
Revision 1.2 - Updated to fix problem with bitmask, now processing correctly

"""

version_num = "1.35"

#############################################
#
# This script requires that the folder specified contains the LogXRF files that
# have been properly uncompressed.  They will have weird names.  This will
# generate updated files with proper CSV extensions.
#
# This script also runs through and plots, per channel, the spectrum and laser
# height + power data.
#
#############################################


#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
# User definitions
# Change these values to modify the behaviour of the output charts
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------


# Select the input path (non-recursive) that has all the CSV files for processing.

pathInputFolder = "YOUR PATH"

# Version of XRFd log file
# version = 3 is the new log format - any other number reverts to old log format
version = 3

# Plot path.  This is the existing folder to which saved plots are written.  If
# savePlots is False, this is not used.
# Use either single forward slashes or double backslashes for the folder separator character.
plotPath = "YOUR PATH"

# Laser threshold to determine which recordings will be considered for health check (in mm)
lasThold = 1000

# Number of channels that the project has
numChannel = 3

# Select the user energy peak to display in the instantaneous peak counts chart.
# If the value is zero, the entire spectrum is treated as a single channel.
# If the energy is 6.40 keV, it will be the Fe Ka peak
userEnergyPeak = 0

# Define a text string to be written with the Y-axis along with the energy peak
# selected above.  For example, "FeKa" would go with 6.40.
userEnergyName = 'All'

# Color of the X-ray high voltage line and Y-axis text
voltageColor = 'red'

# Color of the X-ray DC current line and Y-axis text
currentColor = 'green'

# Color of the selected-peak (given by userEnergyPeak), instantaneous shots
# and Y-axis text
selPeakColor = 'blue'

# Color of the laser height and Y-axis text
laserHtColor = 'purple'

# Color of the spectrum vs energy line.
totSpecColor = 'black'

# Color of the vertical line shown in the spectrum vs energy line for the Fe Ka
# line at 6.40 keV.  Used as a visual aid.
vertFeKaColor = 'fuchsia'
vertCuKaColor = 'fuchsia'

# Color of the vertical dashed line shown in the spectrum vs energy line
# for the Mo Ka line at 17.48 keV.  Used as a visual aid.
vertMoKaColor = 'fuchsia'

# minimum run time in seconds.  If run is less than this time, it is
# noted in text on the chart
minRunTime = 3.0

# Save the plots.  If True, each channel plot is saved as a PDF file.
savePlots = True


#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
# End of User definitions
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------


import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator

import numpy as np
import os.path
import glob
import copy
import pandas as pd
import threading

from datetime import datetime
import time


#############################################
# Async sleep
#############################################
class asyncSleep(threading.Thread):
	def __init__(self, waitSec):
		threading.Thread.__init__(self)
		self.waitSecTime = waitSec


	def run( self ):
		time.sleep( self.waitSecTime )
#############################################
# Find the bin number for a given energy
#############################################
def GetBinNumber( energy, mSlope, bIntercept ):
    binVal = int( round( (energy - bIntercept) / mSlope ) )

    return binVal


#############################################
# Print a Name, Value pair to a single line in a chart
#############################################
def TablePrint( axes, xPos, yPos, title, value, bold=False ):
	xPos2 = float( xPos ) / 100.0
	yPos2 = float( yPos ) / 100.0

	strBold = "normal"
	if bold == True:
		strBold = "bold"
	if title == "":
		axes.text( xPos2, yPos2, '{0}'.format( value ), fontweight=strBold )
	else:
		axes.text( xPos2, yPos2, '{0}: {1}'.format( title, value ), fontweight=strBold )
		
def TablePrint1( axes, xPos, yPos, title, value, bold=False ):
	xPos2 = float( xPos ) / 100.0
	yPos2 = float( yPos ) / 100.0

	if float(value) != 0:
		axes.text( xPos2, yPos2, '{0}: {1}'.format( title, value ), color='red' )
	else:
		axes.text( xPos2, yPos2, '{0}: {1}'.format( title, value ), color='black' )

#############################################
# Print a Name, Value pair
# The name goes to the first chart, the value goes to the second chart
#############################################
def TablePrint2( axes1, axes2, xPosTitle, xPosValue, yPos, title, value, strValue, bold=False, ifGreaterEq=False, cutoffVal=0.0):
	xPos2T = float( xPosTitle ) / 100.0
	xPos2V = float( xPosValue ) / 100.0
	yPos2 = float( yPos ) / 100.0

	bMakeRed = False
	if bold == True:
		if ( ( ( value >= cutoffVal ) and ( ifGreaterEq == True ) ) or
			( ( value <= cutoffVal ) and ( ifGreaterEq == False ) ) ):
			bMakeRed = True

	if bMakeRed == True:
		axes2.text( xPos2V, yPos2, '{0}'.format( strValue ), fontweight="bold", color='red' )
	else:
		axes2.text( xPos2V, yPos2, '{0}'.format( strValue ) )


	axes1.text( xPos2T, yPos2, '{0}'.format( title ), horizontalalignment='right' )



#############################################
# Return the number of entries in a vector that has a bitmask on
# 1 = good
# 0 = bad
#############################################
def HealthCheckCount( healthVector, bitIndex ):
	bitmask = np.power(2, bitIndex)
	outCheck = np.array( [ ( int(c) & bitmask ) for c in healthVector ] )
	mask = outCheck > 0
	numGood = len( outCheck[mask] )
	numBad = len( outCheck ) - numGood
	return numGood, numBad

#############################################
# Return cropped HealthCheck vector based on laser height
# valid data is everything below LasThold
# returns cropped health vector if laserVector varies
# crop is from when scoop starts (based on laser data) to 5 recordings before the last one
# else, returns original health vector
# ###########################################
def healthCheckCrop( laserVector, healthVector, timeVector, lasThold):
	# TODO should not be loading the config file again, should be passing it in -Matt
	#lasMax = np.amax( laserVector )

	lasMin = np.amin( laserVector )
	cropIdx = np.where( laserVector < lasThold )
	
	if lasMin > 0 and len(cropIdx[0]) > 20:
		
		cropIdx_bothEnds = cropIdx[0][0:(len(cropIdx)-5)].copy()
		healthVector_crop = healthVector[cropIdx_bothEnds]
		
		nBot = timeVector[cropIdx_bothEnds[0]]
		nTop = timeVector[cropIdx_bothEnds[-1]]
		
		return healthVector_crop, nBot, nTop 
	else:
		nBot = timeVector[0]
		nTop = timeVector[-1]
		
		return healthVector, nBot, nTop

#############################################
# Clear the ticks in a chart.
#############################################
def ClearTicks( ax ):
	xLabs = ax.get_xticklabels()
	xOutLabs = ['']*len( xLabs )
	ax.set_xticklabels( xOutLabs )
	ax.get_xaxis().set_ticks([])
	yLabs = ax.get_yticklabels()
	yOutLabs = ['']*len( yLabs )
	ax.set_yticklabels( yOutLabs )
	ax.get_yaxis().set_ticks([])

#------------------------------------------------------------------------
# Definitions for files, folder
#------------------------------------------------------------------------


# define a "zero" time for all other times to be based from in terms of elapsed time.
refTime = datetime(2015, 1, 1, 0, 0, 0)

# define dictionary that will store check values for Spec, Volt, Cur and Temp
results = {}

# Go through and rename all files to .csv if needed
fNames = glob.glob( pathInputFolder + "/LogXRF*.*csv*" )
for nIdx in range (0, len(fNames) ):
	fName = fNames[nIdx]
	folder = os.path.dirname(fName)
	base = os.path.basename(fName)
	parts = base.split(".")

	first = parts[0]
	extension = "csv"
	
	newName = first + "." + extension
	newPath = os.path.join(folder, newName)
	os.rename( fName, newPath )



# Parse all files
fNames = glob.glob( pathInputFolder + "/LogXRF*.*csv*" )
filesOrdered = []

for nIdx in range (0, len(fNames) ):
	fName = fNames[nIdx]
	folder = os.path.dirname(fName)
	base = os.path.basename(fName)
	parts = base.split(".")
	if len( parts ) >= 2:
		first = parts[0]
		extension = parts[1]
		newName = first + "." + extension
		newPath = os.path.join(folder, newName)
		os.rename( fName, newPath )

	indivBits = parts[0].split("_")
	if len( indivBits ) > 6:
		print( "    ######################!!!!!!!!!!!!!!!!!!!!!" )
		print( "    ######################" )
		print( "    ######################" )
		print( "    POSSIBLE ERROR: too many underscores in filename.")
		print( "    If you see this script crash, shortly after this")
		print( "    message, this is likely the cause.")
		print( "( {0} )".format( base ) )
		print( "    ######################" )
		print( "    ######################" )
		print( "    ######################!!!!!!!!!!!!!!!!!!!!!" )
	runNum = int( indivBits[3] )
	ymdStr = indivBits[4]
	hmsStr = indivBits[5]
	year = int( ymdStr[0:4] )
	month = int( ymdStr[4:6] )
	dateDay = int( ymdStr[-2:] )

	hours = int( hmsStr[0:2] )
	minutes = int( hmsStr[2:4] )
	seconds = int( hmsStr[-2:] )

	if year == 0:
		year = 2000
	if month == 0:
		month = 1
	if dateDay == 0:
		dateDay = 1


	fileTime = datetime( year, month, dateDay, hours, minutes, seconds )
	diffSecs = (fileTime - refTime).total_seconds()

	# add to a struct that has the filename and ordering values
	filePack = [fName, diffSecs, runNum]
	filesOrdered.append( filePack )


filesOrdered.sort(key=lambda x: ( x[1], x[2] ) )

for nIdx in range (0, len(filesOrdered) ):
	print filesOrdered[nIdx][0]



oneRow = []
xTimeDiffSec = 0.0
minOnPower = 48.0
for nIdx in range (0, len(filesOrdered) ):
	fName = filesOrdered[nIdx][0]
	base = os.path.basename( fName )
	namePart = os.path.splitext( base )[0]
	# read in the dataframe
	df = pd.read_csv( fName, sep=',', dtype=object )
	print( "================================================================================================")

	for nChIdx in range (0, numChannel):
		# channel number is 1-based, channel index in zerobased
		nChNum = nChIdx + 1

		# filter on channel number
		numStr = "{0:d}".format( nChNum )
		dfCh = df[df['Channel'] == numStr]

		if len( dfCh ) < 1:
			print "++++++++++++++++++++++++++++++"
			print "This file ({0}) has no meaningful data for channel {1}.".format( \
				base, numStr )
			print "++++++++++++++++++++++++++++++"
			continue
		missingCal = False
		bin640 = int( np.array( dfCh['Calibration Peak 1'] )[0] )
		bin1748 = int( np.array( dfCh['Calibration Peak 2'] )[0] )
		if bin640 == 0:
			bin640 = 280
			missingCal = True
		if bin1748 == 0:
			bin1748 = 770
			missingCal = True


		xBinAxis = np.arange( 0, 1024 )
		mSlope = ( 17.48 - 6.4 ) / ( bin1748 - bin640 )
		bIntercept = 17.48 - ( mSlope * bin1748 )
		energy = ( xBinAxis * mSlope ) + bIntercept

		binUser = GetBinNumber( userEnergyPeak, mSlope, bIntercept )

		xTmp = dfCh.ix[:, 'Timebase' ]
		xTimebase = np.array( [int(x) for x in xTmp] )
		xTimebase -= xTimebase[0]
		xTimebaseSec = xTimebase / 1000.0

		xDiffTmp = xTimebase[1:] - xTimebase[0:-1]
		xTimeDiff = np.zeros_like( xTimebase )

		lhTmp = dfCh.ix[:, 'Laser Height' ]
		laserHt = np.array( [int(h) for h in lhTmp] )

		kVTmp = dfCh.ix[:, 'kV Feedback' ]
		uATmp = dfCh.ix[:, 'uA Feedback' ]
		kVfb = np.array( [float(v) for v in kVTmp] )
		uAfb = np.array( [float(a) for a in uATmp] )

		power = ( kVfb * uAfb ) / 1000.0

		if (len( xTimeDiff ) > 1 ):
			xTimeDiff[1:] = xDiffTmp
			xTimeDiff[0] = xDiffTmp[0]
			xTimeDiffSec = [float(t)/1000.0 for t in xTimeDiff ]
		else:
			xTimeDiffSec = [ float( xTimeDiff[0] ) / 1000.0 ]

		sumRow = np.zeros_like( energy )

		timeSum = 0.0
		for rowIdx in range( 0, len(xTimebase ) ):
			fullRow = dfCh.iloc[rowIdx]
			tmpRow = fullRow.ix[ 'Bin 0':'Bin 1023' ]
			oneRow = np.array( [float(y) for y in tmpRow] )
			if power[rowIdx] > minOnPower:
				sumRow += oneRow
				timeSum += xTimeDiffSec[rowIdx]

		accumCPS = sumRow / timeSum


		startCol = 0
		endCol = 1023
		startStr= "Bin {0:d}".format( startCol )
		endStr = "Bin {0:d}".format( endCol )

		peakUser = np.zeros_like( xTimebase )
		if userEnergyPeak > 0.0:
			startCol = binUser - 10
			endCol = binUser + 10 + 1
			if endCol > 1023:
				endCol = 1023
			startStr= "Bin {0:d}".format( startCol )
			endStr = "Bin {0:d}".format( endCol )

		for rowIdx in range( 0, len( xTimebase ) ):
			fullRow = dfCh.iloc[rowIdx]
			tmpRowUser = fullRow.ix[ startStr:endStr ]
			oneRowUser = np.array( [float(y) for y in tmpRowUser] )
			peakUser[rowIdx] = np.sum( oneRowUser )


		#++++++++++++++++++++++++++++++++++++++
		#++++++++++++++++++++++++++++++++++++++
		# Tabular Data
		# Display table data here
		#++++++++++++++++++++++++++++++++++++++
		#++++++++++++++++++++++++++++++++++++++


		# get the serial number.  There is the possibility that the column is blank.
		# If blank, just display UNKNOWN.  Since serial number doesn't change, just grab
		# the first row.

		snColTmp = dfCh.ix[:,'Serial Number']
		serialNum = "UNKNOWN"
		if len(snColTmp) > 0:
			serialNum = snColTmp.iloc[0]


		# get the temperature.  There is the possibility that the column is blank.
		if version == 2:
			# get the temperature.  There is the possibility that the column is blank.
			temperColTmp = dfCh.ix[:,'Temperature']
			temperCol = temperColTmp.astype( "float" )
			#temperCol = np.array( [ float(t) for t in temperColTmp ] )
			detectorTemperatureMin = 999
			detectorTemperatureMax = 999
			detectorTemperatureAvg = 999
			if len(temperCol) > 0:
				detectorTemperatureAvg = np.average( temperCol )
				detectorTemperatureMin = np.min( temperCol )
				detectorTemperatureMax = np.max( temperCol )			
		elif version == 3:
			# get the temperature.  There is the possibility that the column is blank.
			temperColTmp = dfCh.ix[:,'Detector Temperature']
			temperCol = temperColTmp.astype( "float" )
			#temperCol = np.array( [ float(t) for t in temperColTmp ] )
			detectorTemperatureMin = 999
			detectorTemperatureMax = 999
			detectorTemperatureAvg = 999
			if len(temperCol) > 0:
				detectorTemperatureAvg = np.average( temperCol )
				detectorTemperatureMin = np.min( temperCol )
				detectorTemperatureMax = np.max( temperCol )			
				
			# get the Emitter temperature.  There is the possibility that the column is blank.
			emitterTemperColTmp = dfCh.ix[:,'Emitter Temperature']
			emitterTemperCol = emitterTemperColTmp.astype( "float" )
			#temperCol = np.array( [ float(t) for t in temperColTmp ] )
			emitterTemperatureMin = 999
			emitterTemperatureMax = 999
			emitterTemperatureAvg = 999
			if len(emitterTemperCol) > 0:
				emitterTemperatureAvg = np.average( emitterTemperCol )
				emitterTemperatureMin = np.min( emitterTemperCol )
				emitterTemperatureMax = np.max( emitterTemperCol )
		else:
			print('**************************************')
			print('**************************************')
			print('**************************************')
			print('**************************************')
			print('set the LOG version!')
			print('**************************************')
			print('**************************************')
			print('**************************************')
			print('**************************************')
			
			quit()

		# get the voltage.
		kVMin = 0

		kVMax = 0
		kVAvg = 0
		if len(kVfb) > 0:
			kvAvg = np.average( kVfb )
			kvMin = np.min( kVfb )
			kvMax = np.max( kVfb )

		# get the current.
		uAMin = 0
		uAMax = 0
		uAAvg = 0
		if len(uAfb) > 0:
			uAAvg = np.average( uAfb )
			uAMin = np.min( uAfb )
			uAMax = np.max( uAfb )

		if version == 2:	
			# get the deadtime.  Convert to percentage
			dtColTmp = dfCh.ix[:,'Detector Dead Time Percentage']
			dtColPerc = np.array( [float(dt) for dt in dtColTmp] )
			dtMin = 0
			dtMax = 0
			dtAvg = 0
			if len(dtColPerc) > 0:
				dtAvg = np.average( dtColPerc )
				dtMin = np.min( dtColPerc )
				dtMax = np.max( dtColPerc )
		elif version == 3:		
			# get the deadtime.  Convert to percentage
			dtICR       = dfCh.ix[:,'Input Count Rate']
			dtICR_array = np.array( [float(dt) for dt in dtICR] )
			dtOCR    = dfCh.ix[:,'Output Count Rate']
			dtOCR_array = np.array( [float(dt) for dt in dtOCR] )
			
			dtColPerc = 100* (1 - dtOCR_array/dtICR_array)
	
			dtMin = 0
			dtMax = 0
			dtAvg = 0
			if len(dtColPerc) > 0:
				dtAvg = np.average( dtColPerc )
				dtMin = np.min( dtColPerc )
				dtMax = np.max( dtColPerc )

		# get the laser height.
		laserHtAvg = 0
		laserHtNearFull = 0
		laserHtNearEmpty = 0
		if len(laserHt) > 0:
			laserHtAvg = np.average( laserHt )
			laserHtNearFull = np.min( laserHt )
			laserHtNearEmpty = np.max( laserHt )

		# get the Health Code
		# get the Health Code
		healthColTmp = dfCh.ix[:,'Health Code']
		healthCol = np.array( [int(h) for h in healthColTmp] )
		
		# use laser data to remove data when NOT scooping
		healthCol_crop, n_bot, n_top = healthCheckCrop( laserHt, healthCol, xTimebaseSec, lasThold )
		nPts = len(healthCol_crop)
			
		# setup checklist: 0 is GOOD, ~= 0 is BAD
		numCurrData, numOldData = HealthCheckCount( healthCol_crop, 0 )
		numGoodSpectrum, numBadSpectrum = HealthCheckCount( healthCol_crop, 1 )
		numGoodVoltage, numBadVoltage = HealthCheckCount( healthCol_crop, 2 )
		numGoodCurrent, numBadCurrent = HealthCheckCount( healthCol_crop, 3 )
		numGoodTemperature, numBadTemperature = HealthCheckCount( healthCol_crop, 4 )
		
		if version == 3:
			# new healthcheck parameters
			numGoodShutter, numBadShutter = HealthCheckCount( healthCol_crop, 5 )
			numGoodShutterHealth, numBadShutterHealth = HealthCheckCount( healthCol_crop, 6 )
			numGoodEmitterTemp, numBadEmitterTemp = HealthCheckCount( healthCol_crop, 7 )
			numGoodRIO, numBadRIO = HealthCheckCount( healthCol_crop, 8 )		
						
		checkData = 100 * float( numOldData ) / nPts	
		checkSpec = 100 * float( numBadSpectrum ) / nPts
		checkVolt = 100 * float( numBadVoltage ) / nPts
		checkCur  = 100 * float( numBadCurrent ) / nPts
		checkTemp = 100 * float( numBadTemperature ) / nPts
		run_name = namePart + '_CH' + str(nChNum)
		results.update({ run_name: [checkData, checkSpec, checkVolt, checkCur, checkTemp] })
		
		if version == 3:
			checkRIO           = 100 * float( numBadRIO ) / nPts
			checkShutter       = 100 * float( numBadShutter ) / nPts
			checkShutterHealth = 100 * float( numBadShutterHealth ) / nPts
			checkEmitterTemp   = 100 * float( numBadEmitterTemp ) / nPts
			results.update({ run_name: [checkData, checkSpec, checkVolt, checkCur, checkTemp, checkRIO, checkShutter, checkShutterHealth, checkEmitterTemp] })
			
		bShortRun = False
		bNeverPoweredXray = False
		bNoAccum = False

		if ( uAMax <= 0.001 ) and ( kVMax <= 0.001 ):
			bNeverPoweredXray = True
		if np.max( xTimebaseSec ) <= minRunTime:
			bShortRun = True
		if np.sum( accumCPS ) <= 0:
			bNoAccum = True

		#++++++++++++++++++++++++++++++++++++++
		#++++++++++++++++++++++++++++++++++++++
		# Plots
		#++++++++++++++++++++++++++++++++++++++
		#++++++++++++++++++++++++++++++++++++++
		if version == 3:
			# make figure with 1 subplot
			fig = plt.figure( figsize = (14, 9) )
			axes = fig.add_subplot( 111 )
		
			#++++++++++++++++++++++++++++++++++++++
			# Plot (1a) - Voltage and Current vs Time
			#++++++++++++++++++++++++++++++++++++++
			ax1a = plt.subplot2grid( (100, 100), (0, 0), rowspan=25, colspan=50 )
		
			plt.plot( xTimebaseSec, kVfb, color=voltageColor, linestyle='-', marker='.', linewidth='1.5', label="Voltage (kV)" )
			ax1a.autoscale()
			ax1a.set_ylim([-8, 60])
			yTickVals = list( np.arange(0, 51, 10) )
			yTickVals.insert( 0, -8 )
			yTickVals.append( 60 )
			yTickVals = np.hstack( np.array( yTickVals ).flat )
			ax1a.set_yticks(yTickVals)
			yTickLabels = [ '', '0', '10', '20', '30', '40', '50', '' ]
			ax1a.set_yticklabels( yTickLabels )
			minorLocator = AutoMinorLocator(10)
			ax1a.yaxis.set_minor_locator(minorLocator)
		
			#ax1a.set_xlabel('Elapsed Time (s)')
			ax1a.set_ylabel('X-ray Volt (kV)')
			ax1a.yaxis.label.set_color( voltageColor )
			ax1a.spines['left'].set_color( voltageColor )
			ax1a.tick_params(axis='y', colors = voltageColor )
			plt.grid(True)
		
			xLabs = ax1a.get_xticklabels()
			xOutLabs = ['']*len( xLabs )
			ax1a.set_xticklabels( xOutLabs )
		
		
			ax1a_2 = ax1a.twinx()
			ax1a_2.set_ylim([-160, 1200])
		
			yTickVals = list( np.arange(0, 1001, 200) )
			yTickVals.insert( 0, -160 )
			yTickVals.append( 1200 )
			yTickVals = np.hstack( np.array( yTickVals ).flat )
			ax1a_2.set_yticks(yTickVals)
			yTickLabels = ['', '0', '200', '400', '600', '800', '1000', '']
			ax1a_2.set_yticklabels( yTickLabels )
		
			ax1a_2.set_ylabel("X-ray Curr (uA)")
		
			plt.plot( xTimebaseSec, uAfb, color=currentColor, linestyle='-', marker='None', linewidth='1.5', label="Current (uA)" )
			ax1a_2.yaxis.label.set_color( currentColor )
			ax1a_2.spines['right'].set_color( currentColor )
			ax1a_2.tick_params(axis='y', colors = currentColor )
		
			if len(healthCol) != len(healthCol_crop):
				ax1a_3 = ax1a.twinx()
				ax1a_3.set_ylim([0, 1])
				ax1a_3.yaxis.set_visible(False)
				plt.plot([n_bot, n_bot],[0, 1], color='black', linestyle='dashed', marker='None', linewidth=1.5)
				
				ax1a_4 = ax1a.twinx()
				ax1a_4.set_ylim([0, 1])
				ax1a_4.yaxis.set_visible(False)
				plt.plot([n_top, n_top],[0, 1], color='black', linestyle='dashed', marker='None', linewidth=1.5)
		
			if bNeverPoweredXray == True:
				xValLow, xValHigh = ax1a.get_xlim()
				xPos = 0.2 * ( xValHigh - xValLow ) + xValLow
				ax1a.text( xPos, 25, "NEVER POWERED UP", fontweight="bold", color='red', fontsize='20' )
		
		
			#++++++++++++++++++++++++++++++++++++++
			# Plot (1b) - Accumulated User-defined Peak, Laser height
			#++++++++++++++++++++++++++++++++++++++
		
			ax1b = plt.subplot2grid( (100, 100), (25, 0), rowspan=25, colspan=50 )
		
			ax1b.yaxis.tick_left()
			ax1b.yaxis.set_label_position("left")
			ax1b.yaxis.label.set_color( selPeakColor )
			ax1b.spines['left'].set_color( selPeakColor )
			ax1b.tick_params(axis='y', colors = selPeakColor )
		
			plt.plot( xTimebaseSec, peakUser, color=selPeakColor, linestyle='-', marker='.', linewidth='1.5', label="Fe Ka" )
		
			ax1b.set_xlabel('Elapsed Time (s)')
			ax1b.set_ylabel('kiloCounts ({0})'.format( userEnergyName ))
			plt.grid(True)
		
		
			ax1b_2 = ax1b.twinx()
			ax1b_2.yaxis.tick_right()
			ax1b_2.yaxis.set_label_position("right")
			ax1b_2.set_ylim([-100, 1200])
			yTickVals = list( np.arange(-100, 1201, 100 ) )
			ax1b_2.set_yticks(yTickVals)
			yTickLabels = [ '', '0.0', '', '0.2', '',
							'0.4', '', '0.6', '', '0.8', '', '1.0', '', ''  ]
			ax1b_2.set_yticklabels( yTickLabels )
		
			ax1b_2.set_ylabel("Laser Height (m)")
		
			plt.plot( xTimebaseSec, laserHt, color=laserHtColor, linestyle='-', marker='.', linewidth='1.5', label="Laser Height" )
			ax1b_2.yaxis.label.set_color( laserHtColor )
			ax1b_2.spines['right'].set_color( laserHtColor )
			ax1b_2.tick_params(axis='y', colors = laserHtColor )
		
			if len(healthCol) != len(healthCol_crop):
				ax1b_3 = ax1b.twinx()
				ax1b_3.set_ylim([0, 1])
				ax1b_3.yaxis.set_visible(False)
				plt.plot([n_bot, n_bot],[0, 1], color='black', linestyle='dashed', marker='None', linewidth=1.5)
				
				ax1b_4 = ax1b.twinx()
				ax1b_4.set_ylim([0, 1])
				ax1b_4.yaxis.set_visible(False)
				plt.plot([n_top, n_top],[0, 1], color='black', linestyle='dashed', marker='None', linewidth=1.5)		
		
			yLow, yHigh = ax1b.get_ylim()
			if yHigh < 200:
				ax1b.set_ylim( [0, 200] )
		
			yLow, yHigh = ax1b.get_ylim()
			yLabs = ax1b.get_yticklabels( False )
		
			diff = ( yHigh - yLow ) /11.0
			ax1b.set_ylim( [yLow - diff, yHigh + diff] )
		
			yTickVals = list( np.arange(yLow, yHigh + 1, ( yHigh - yLow )/( len(yLabs) - 1 ) ) )
			yTickValsOrig = copy.copy( yTickVals )
		
			yLabs = ax1b.get_yticklabels( False )
			# numYTicks = len( yLabs ) - 2
			yOut2 = [ float(x / 1000) for x in yTickValsOrig ]
		
			yLabsNew = list( [ "{0:.2f}".format( x ) for x in yOut2 ] )
			yLabsNew.insert( 0, "" )
			yLabsNew.append( "" )
			ax1b.set_yticklabels( yLabsNew )
		
			if bShortRun == True:
				xValLow, xValHigh = ax1b.get_xlim()
				xPos = 0.2 * ( xValHigh - xValLow ) + xValLow
				yValLow, yValHigh = ax1b.get_ylim()
				yPos = 0.5 * ( yValHigh - yValLow ) + yValLow
				ax1b.text( xPos, yPos, "SHORT RUN <={0:.1f} s".format( configInfo.minRunTimeSec ), fontweight="bold", color='blue', fontsize='20' )
		
		
			#++++++++++++++++++++++++++++++++++++++
			# Plot (2) - Status Text area
			#++++++++++++++++++++++++++++++++++++++
		
			# Title area
			ax2a = plt.subplot2grid( (100, 100), (0, 57), rowspan=8, colspan=43 )
			ax2b = plt.subplot2grid( (100, 100), (8, 57), rowspan=48, colspan=24 )
			ax2c = plt.subplot2grid( (100, 100), (8, 81), rowspan=48, colspan=19 )
		
			ClearTicks( ax2a )
			ClearTicks( ax2b )
			ClearTicks( ax2c )
		
			TablePrint( ax2a, 1, 72, "", namePart, True )
			TablePrint( ax2a, 1, 38, "Channel Number", numStr, True )
			TablePrint( ax2a, 1, 5, "Serial Number", serialNum, True )
		
			xIndexTitle = 98
			xIndexVal = 1
			yIndex = 93
			ySpacing = 8
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Det. Temp ($^\circ$C) (min,avg,max)", 0, "{0:.3f},{1:.3f},{2:.3f}".format( detectorTemperatureMin, detectorTemperatureAvg, detectorTemperatureMax ) )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Em. Temp ($^\circ$C) (min,avg,max)", 0, "{0:.0f},  {1:.0f},  {2:.0f}".format( emitterTemperatureMin, emitterTemperatureAvg, emitterTemperatureMax ) )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Deadtime (%) (min,avg,max)", 0, "{0:.1f},  {1:.1f},  {2:.1f}".format( dtMin, dtAvg, dtMax ) )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "LaserHt (mm) (full,avg,empty)", 0, "{0:.0f}, {1:.0f}, {2:.0f}".format( laserHtNearFull, laserHtAvg, laserHtNearEmpty ) )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Total On-time (s) and # of Pts", 0, "{0:.3f}, {1:d}".format( timeSum, nPts )  )
			yIndex -= ySpacing
			TablePrint1( ax2b, xIndexVal, yIndex, "bad Shutter Health (%)", "{0:.1f}".format( checkShutterHealth ) )
			TablePrint1( ax2c, xIndexVal, yIndex, "bad Shutter (%)", "{0:.1f}".format( checkShutter ) )
			yIndex -= ySpacing
			TablePrint1( ax2b, xIndexVal, yIndex, "bad Spectrum (%)", "{0:.1f}".format( checkSpec ) )
			TablePrint1( ax2c, xIndexVal, yIndex, "bad Packet (%)", "{0:.1f}".format( checkData ) )
			yIndex -= ySpacing
			TablePrint1( ax2b, xIndexVal, yIndex, "bad Voltage (%)", "{0:.1f}".format( checkVolt ) )
			TablePrint1( ax2c, xIndexVal, yIndex, "bad Det. Temp (%)", "{0:.1f}".format( checkTemp ) )
			yIndex -= ySpacing
			TablePrint1( ax2b, xIndexVal, yIndex, "bad DC Curr. (%)", "{0:.1f}".format( checkCur ) )
			TablePrint1( ax2c, xIndexVal, yIndex, "bad RIO (%)", "{0:.1f}".format( checkRIO ) )
			yIndex -= ySpacing
			TablePrint1( ax2b, xIndexVal, yIndex, "bad Emitter Temp (%)", "{0:.1f}".format( checkEmitterTemp ) )
		
			#++++++++++++++++++++++++++++++++++++++
			# Plot (3) - Cumulative Final Counts
			#++++++++++++++++++++++++++++++++++++++
		
			ax3 = plt.subplot2grid( (100, 100), (57, 0), rowspan=43, colspan=100 )
		
		
			plt.plot( energy, accumCPS, color=totSpecColor, linestyle='-', marker='None', linewidth='3.5', )
		
			ax3.set_xlim([0, 23])
			xTickVals = np.arange(0, 24)
			ax3.set_xticks(xTickVals)
			minorLocator = AutoMinorLocator(10)
			axes.xaxis.set_minor_locator(minorLocator)
		
			yLow, yHigh = axes.get_ylim()
			if yHigh < 800:
				ax3.set_ylim( [0, 800] )
		
			yLow, yHigh = ax3.get_ylim()
			plt.plot( [6.4, 6.4], [yLow, yHigh], color=vertFeKaColor, linestyle='-', marker='None', linewidth=2.5, label='Fe Ka Peak' )
			str1 = '(%d)' % bin640
			str2 = '(%d)' % bin1748
			plt.text( 6.5, 0.9*yHigh, str1, color=vertFeKaColor )
			plt.plot( [17.48, 17.48], [yLow, yHigh], color=vertCuKaColor, linestyle='--', marker='None', linewidth=2.5, label='Mo Ka Peak' )
			plt.text( 17.6, 0.9*yHigh, str2, color=vertCuKaColor )
		
			ax3.set_xlabel('Energy (keV)')
			ax3.set_ylabel('Total Run CPS')
			plt.grid(True)
		
			if bNoAccum == True:
				xValLow, xValHigh = ax3.get_xlim()
				xPos = 0.2 * ( xValHigh - xValLow ) + xValLow
				yValLow, yValHigh = ax3.get_ylim()
				yPos = 0.5 * ( yValHigh - yValLow ) + yValLow
				ax3.text( xPos, yPos, "NO DATA ACCUMULATED", fontweight="bold", color='brown', fontsize='20' )
		
			if missingCal == True:
				xValLow, xValHigh = ax3.get_xlim()
				xPos = 0.4 * ( xValHigh - xValLow ) + xValLow
				yValLow, yValHigh = ax3.get_ylim()
				yPos = 0.8 * ( yValHigh - yValLow ) + yValLow
				ax3.text( xPos, yPos, "MISSING CALIBRATION", fontweight="bold", color='brown', fontsize='20' )			
			
			
		else:
			fig = plt.figure( figsize = (14, 9) )
			#plt.ion()
			#plt.cla()
			#plt.clf()
	
			# define one large plot area
			axes = fig.add_subplot( 111 )
	
			# Make grid of 20x20 in size, start top left chart with voltage and current
	
			#++++++++++++++++++++++++++++++++++++++
			# Plot (1a) - Voltage and Current vs Time
			#++++++++++++++++++++++++++++++++++++++
			ax1a = plt.subplot2grid( (100, 100), (0, 0), rowspan=25, colspan=50 )
	
			plt.plot( xTimebaseSec, kVfb, color=voltageColor, linestyle='-', marker='.', linewidth=1.5, label="Voltage (kV)" )
			ax1a.autoscale()
			ax1a.set_ylim([-8, 60])
			yTickVals = list( np.arange(0, 51, 10) )
			yTickVals.insert( 0, -8 )
			yTickVals.append( 60 )
			yTickVals = np.hstack( np.array( yTickVals ).flat )
			ax1a.set_yticks(yTickVals)
			yTickLabels = [ '', '0', '10', '20', '30', '40', '50', '' ]
			ax1a.set_yticklabels( yTickLabels )
			minorLocator = AutoMinorLocator(10)
			ax1a.yaxis.set_minor_locator(minorLocator)
	
			#ax1a.set_xlabel('Elapsed Time (s)')
			ax1a.set_ylabel('X-ray Volt (kV)')
			ax1a.yaxis.label.set_color( voltageColor )
			ax1a.spines['left'].set_color( voltageColor )
			ax1a.tick_params(axis='y', colors = voltageColor )
			plt.grid(True)
	
			xLabs = ax1a.get_xticklabels()
			xOutLabs = ['']*len( xLabs )
			ax1a.set_xticklabels( xOutLabs )
	
	
			ax1a_2 = ax1a.twinx()
			ax1a_2.set_ylim([-160, 1200])
	
			yTickVals = list( np.arange(0, 1001, 200) )
			yTickVals.insert( 0, -160 )
			yTickVals.append( 1200 )
			yTickVals = np.hstack( np.array( yTickVals ).flat )
			ax1a_2.set_yticks(yTickVals)
			yTickLabels = ['', '0', '200', '400', '600', '800', '1000', '']
			ax1a_2.set_yticklabels( yTickLabels )
	
			ax1a_2.set_ylabel("X-ray Curr (uA)")
	
			plt.plot( xTimebaseSec, uAfb, color=currentColor, linestyle='-', marker='None', linewidth=1.5, label="Current (uA)" )
			ax1a_2.yaxis.label.set_color( currentColor )
			ax1a_2.spines['right'].set_color( currentColor )
			ax1a_2.tick_params(axis='y', colors = currentColor )
			
			if timeSum > 2:
				ax1a_3 = ax1a.twinx()
				ax1a_3.set_ylim([0, 1])
				ax1a_3.yaxis.set_visible(False)
				plt.plot([n_bot, n_bot],[0, 1], color='black', linestyle='dashed', marker='None', linewidth=1.5)
	
				ax1a_4 = ax1a.twinx()
				ax1a_4.set_ylim([0, 1])
				ax1a_4.yaxis.set_visible(False)
				plt.plot([n_top, n_top],[0, 1], color='black', linestyle='dashed', marker='None', linewidth=1.5)
	
			if bNeverPoweredXray == True:
				xValLow, xValHigh = ax1a.get_xlim()
				xPos = 0.2 * ( xValHigh - xValLow ) + xValLow
				ax1a.text( xPos, 25, "NEVER POWERED UP", fontweight="bold", color='red', fontsize='20' )
	
	
			#++++++++++++++++++++++++++++++++++++++
			# Plot (1b) - Accumulated User-defined Peak, Laser height
			#++++++++++++++++++++++++++++++++++++++
	
			ax1b = plt.subplot2grid( (100, 100), (25, 0), rowspan=25, colspan=50 )
	
			ax1b.yaxis.tick_left()
			ax1b.yaxis.set_label_position("left")
			ax1b.yaxis.label.set_color( selPeakColor )
			ax1b.spines['left'].set_color( selPeakColor )
			ax1b.tick_params(axis='y', colors = selPeakColor )
	
			plt.plot( xTimebaseSec, peakUser, color=selPeakColor, linestyle='-', marker='.', linewidth=1.5, label="Fe Ka" )
	
			ax1b.set_xlabel('Elapsed Time (s)')
			ax1b.set_ylabel('kiloCounts ({0})'.format( userEnergyName ))
			plt.grid(True)
	
	
			ax1b_2 = ax1b.twinx()
			ax1b_2.yaxis.tick_right()
			ax1b_2.yaxis.set_label_position("right")
			ax1b_2.set_ylim([-100, 1200])
			yTickVals = list( np.arange(-100, 1201, 100 ) )
			ax1b_2.set_yticks(yTickVals)
			yTickLabels = [ '', '0.0', '', '0.2', '', \
							'0.4', '', '0.6', '', '0.8', '', '1.0', '', ''  ]
			ax1b_2.set_yticklabels( yTickLabels )
	
			ax1b_2.set_ylabel("Laser Height (m)")
	
			plt.plot( xTimebaseSec, laserHt, color=laserHtColor, linestyle='-', marker='.', linewidth=1.5, label="Laser Height" )
			ax1b_2.yaxis.label.set_color( laserHtColor )
			ax1b_2.spines['right'].set_color( laserHtColor )
			ax1b_2.tick_params(axis='y', colors = laserHtColor )
	
	
			yLow, yHigh = ax1b.get_ylim()
			if yHigh < 200:
				ax1b.set_ylim( [0, 200] )
	
			yLow, yHigh = ax1b.get_ylim()
			yLabs = ax1b.get_yticklabels( False )
	
			diff = ( yHigh - yLow ) /11.0
			ax1b.set_ylim( [yLow - diff, yHigh + diff] )
	
			#yTickVals = list( np.arange(yLow, yHigh + 1, ( yHigh - yLow )/( len(yLabs) - 1 ) ) )
			yTickVals = list( np.arange(yLow, yHigh + 1, ( yHigh - yLow )/4 ) )
			yTickValsOrig = copy.copy( yTickVals )
	
			yLabs = ax1b.get_yticklabels( False )
			#numYTicks = len( yLabs ) - 2
			numYTicks = 5
			yOut2 = [ float(x / 1000) for x in yTickValsOrig ]
	
			yLabsNew = list( [ "{0:.2f}".format( x ) for x in yOut2 ] )
			yLabsNew.insert( 0, "" )
			yLabsNew.append( "" )
			ax1b.set_yticklabels( yLabsNew )
	
			if timeSum > 2:
				ax1b_3 = ax1b.twinx()
				ax1b_3.set_ylim([0, 1])
				ax1b_3.yaxis.set_visible(False)
				plt.plot([n_bot, n_bot],[0, 1], color='black', linestyle='dashed', marker='None', linewidth=1.5)
	
				ax1b_4 = ax1b.twinx()
				ax1b_4.set_ylim([0, 1])
				ax1b_4.yaxis.set_visible(False)
				plt.plot([n_top, n_top],[0, 1], color='black', linestyle='dashed', marker='None', linewidth=1.5)
	
			if bShortRun == True:
				xValLow, xValHigh = ax1b.get_xlim()
				xPos = 0.2 * ( xValHigh - xValLow ) + xValLow
				yValLow, yValHigh = ax1b.get_ylim()
				yPos = 0.5 * ( yValHigh - yValLow ) + yValLow
				ax1b.text( xPos, yPos, "SHORT RUN <={0:.1f} s".format( minRunTime ), fontweight="bold", color='blue', fontsize='20' )
	
	
			#++++++++++++++++++++++++++++++++++++++
			# Plot (2) - Status Text area
			#++++++++++++++++++++++++++++++++++++++
	
			# Title area
			ax2a = plt.subplot2grid( (100, 100), (0, 57), rowspan=8, colspan=43 )
			ax2b = plt.subplot2grid( (100, 100), (8, 57), rowspan=48, colspan=24 )
			ax2c = plt.subplot2grid( (100, 100), (8, 81), rowspan=48, colspan=19 )
	
			ClearTicks( ax2a )
			ClearTicks( ax2b )
			ClearTicks( ax2c )
	
			TablePrint( ax2a, 1, 72, "", namePart, True )
			TablePrint( ax2a, 1, 38, "Channel Number", numStr, True )
			TablePrint( ax2a, 1, 5, "Serial Number", serialNum, True )
	
			xIndexTitle = 98
			xIndexVal = 1
			yIndex = 93
			ySpacing = 8
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Det Temp ($^\circ$C) (min,avg,max)", 0, "{0:.3f},{1:.3f},{2:.3f}".format( detectorTemperatureMin, detectorTemperatureAvg, detectorTemperatureMax ) )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Voltage (kv) (min,avg,max)", 0, "{0:.0f},  {1:.0f},  {2:.0f}".format( kvMin, kvAvg, kvMax ) )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Current (uA) (min,avg,max)", 0, "{0:.0f},  {1:.0f},  {2:.0f}".format( uAMin, uAAvg, uAMax ) )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Deadtime (%) (min,avg,max)", 0, "{0:.2f},  {1:.2f},  {2:.2f}".format( dtMin, dtAvg, dtMax ) )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "LaserHt (mm) (full,avg,empty)", 0, "{0:.0f}, {1:.0f}, {2:.0f}".format( laserHtNearFull, laserHtAvg, laserHtNearEmpty ) )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Calibration bins (FeKa, MoKa)", 0, "{0:d}, {1:d}".format( bin640, bin1748 ) )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Total On-time (s)", timeSum, "{0:.3f}".format( timeSum )  )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Packet OK count (good, bad)", numOldData, "{0:d}, {1:d}".format( numCurrData, numOldData ), True, True, 1 )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Spectrum count (good, bad)", numBadSpectrum, "{0:d}, {1:d}".format( numGoodSpectrum, numBadSpectrum ), True, True, 1 )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Voltage count (good, bad)", numBadVoltage, "{0:d}, {1:d}".format( numGoodVoltage, numBadVoltage ), True, True, 1 )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "DC Curr. count (good, bad)", numBadCurrent, "{0:d}, {1:d}".format( numGoodCurrent, numBadCurrent ), True, True, 1 )
			yIndex -= ySpacing
			TablePrint2( ax2b, ax2c, xIndexTitle, xIndexVal, yIndex, "Temper. count (good, bad)", numBadTemperature, "{0:d}, {1:d}".format( numGoodTemperature, numBadTemperature ), True, True, 1 )
	
			#++++++++++++++++++++++++++++++++++++++
			# Plot (3) - Cumulative Final Counts
			#++++++++++++++++++++++++++++++++++++++
	
			ax3 = plt.subplot2grid( (100, 100), (57, 0), rowspan=43, colspan=100 )
	
	
			plt.plot( energy, accumCPS, color=totSpecColor, linestyle='-', marker='None', linewidth='2.5', )
	
			ax3.set_xlim([0, 23])
			xTickVals = np.arange(0, 24)
			ax3.set_xticks(xTickVals)
			minorLocator = AutoMinorLocator(10)
			axes.xaxis.set_minor_locator(minorLocator)
	
			yLow, yHigh = ax3.get_ylim()
			if yHigh < 800:
				ax3.set_ylim( [0, 100] )
	
			yLow, yHigh = ax3.get_ylim()
			plt.plot( [6.4, 6.4], [yLow, yHigh], color=vertFeKaColor, linestyle='-', marker='None', linewidth=2.5, label='Fe Ka Peak' )
			plt.plot( [17.48, 17.48], [yLow, yHigh], color=vertMoKaColor, linestyle='--', marker='None', linewidth=2.5, label='Mo Ka Peak' )
	
			ax3.set_xlabel('Energy (keV)       [this analyzer script version is v{0}]'.format( version_num ))
			ax3.set_ylabel('Total Run CPS')
			plt.grid(True)
	
			if bNoAccum == True:
				xValLow, xValHigh = ax3.get_xlim()
				xPos = 0.2 * ( xValHigh - xValLow ) + xValLow
				yValLow, yValHigh = ax3.get_ylim()
				yPos = 0.5 * ( yValHigh - yValLow ) + yValLow
				ax3.text( xPos, yPos, "NO DATA ACCUMULATED", fontweight="bold", color='brown', fontsize='20' )
	
			if missingCal == True:
				xValLow, xValHigh = ax3.get_xlim()
				xPos = 0.4 * ( xValHigh - xValLow ) + xValLow
				yValLow, yValHigh = ax3.get_ylim()
				yPos = 0.8 * ( yValHigh - yValLow ) + yValLow
				ax3.text( xPos, yPos, "MISSING CALIBRATION", fontweight="bold", color='brown', fontsize='20' )
	
			#plt.hold( False )
			#plt.show()
			plt.draw()
			#plt.pause( 1e-16 )
	
		#++++++++++++++++++++++++++++++++++++++
		# Possibly save the (single-page) plot with 3 charts and one table
		#++++++++++++++++++++++++++++++++++++++
		if savePlots == True:
			figName = plotPath + "/" + namePart + "-ch{0}".format(numStr) + '.png'
			plt.savefig( figName )
		 	plt.close()

pd.set_option('display.precision',3)
if version == 3:
	df_results = pd.DataFrame(results, index = ['Packets Dropped (% Bad)', 'Spectrum Counts (% Bad)', 'Feedback Voltage (% Bad)', 'Feedback Current (% Bad)', 'Sensor Temperature (% Bad)', 'Sensor RIO (% Bad)', 'Sensor Shutter (% Bad)', 'Sensor Shutter Health (% Bad)', 'Sensor Emitter Temperature (% Bad)'] )
elif version == 2:
	df_results = pd.DataFrame(results, index = ['Packets Dropped (% Bad)', 'Spectrum Counts (% Bad)', 'Feedback Voltage (% Bad)', 'Feedback Current (% Bad)', 'Sensor Temperature (% Bad)' ] )
	
df_results = df_results.transpose()

check_location = plotPath + "/" + 'checklist.csv'
df_results.to_csv(check_location)