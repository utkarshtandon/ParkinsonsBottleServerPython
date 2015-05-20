from __future__ import division
from bottle import route, run, template, post, request
import csv
import os
import numpy
import pyfits
import numpy as np
from scipy.fftpack import fft, rfft, fftfreq
import pylab as plt
import pyimgur
import pickle
from sklearn.ensemble import RandomForestClassifier

CLIENT_ID = "f0e83904a2dee5a"
print "worked till now"


xlist = []
ylist = []
peakvalues_dict = dict()
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        stuff_to_return = _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    #return ind
    return stuff_to_return
    


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            xvalue = (ind)
            yvalue = x[ind]
            for i in xvalue:
            	xlist.append(i)
            for i in yvalue:
            	i = round(i, 5)
            	ylist.append(i)
            peakvalues_dict = dict(zip(xlist, ylist))
            xlist[:] = []
            ylist[:] = []
            satipo = 1
            ax.legend(loc='best', numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('time', fontsize=14)
        ax.set_ylabel('Acceleration Values', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        #plt.show()
        #if valley == False:
        	#save("image/amplitude", ext="png", close=True, verbose=True)
        return peakvalues_dict


def save(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.

    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.

    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.

    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.

    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.

    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'
 
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
 
    # The final path to save to
    savepath = os.path.join(directory, filename)
 
    if verbose:
        print("Saving figure to '%s'..." % savepath),
 
    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()
 
    if verbose:
        print("Done")

@route('/api', method='POST')
def index():
	stringfromapp = request.POST.get('string').strip()
	text_file = open("Output.txt", "w")
	text_file.write(stringfromapp)
	text_file.close()
	inputarray= []
	fo=open("Output.txt","r")
	for line in fo:
		inputarray.append(line.rstrip("\n"))
	fo.close()
	inputarray=list(inputarray)
	#print inputarray
	array=()
	counter=0
	#Over here we write the values from the text file into a properly formatted csv file
	data1 = []
	c = csv.writer(open("data.csv", "wb"))
	for n in inputarray:
		counter=counter+1
		if counter==1:
			t=n
		if counter==2:
			x=n
		if counter==3:
			y=n
		if counter==4:
			z=n
			array = (t,x,y,z)
			#print array[1]
			data1.append(array[1])
			c.writerow(array)
			array=()
			counter=0
	#Now we open the csv and take its values	
	data1 = []
	file = open("data.csv")
	reader = csv.reader(file)
	# print reader
	for line in reader:
		#print line[0]
		try:
			data1.append(line[1])
		except IndexError:
			continue
	# print data1
	tdata1 = np.array(data1, dtype='float')

	#data1 = numpy.genfromtxt("data.csv",dtype='float',delimiter =
	#',',skiprows=0, skip_header=0, skip_footer=0,
	                        #usecols = (1),usemask=True)
	print "Recieved"
	y = tdata1
	amount = len(tdata1)
	datax = []
	xvalue=0
	for n in range(0, amount):
	    xvalue=xvalue+1
	    datax.append(xvalue)
	x = datax
	#x,y = np.loadtxt('fouriertext.txt', usecols = (0,1), unpack=True)
	y = y - y.mean()

	W = fftfreq(y.size, d=(0.02))
	'''
	plt.subplot(2,1,1)
	plt.plot(x,y)
	plt.xlabel('Time')
	'''
	f_signal = fft(y)
	#plt.subplot(2,1,2)
	plt.plot(W, abs(f_signal)**2)
	plt.xlabel('Frequencies (Hz)')
	plt.ylabel('Power')



	max_y = max(abs(f_signal)**2)  # Find the maximum y value
	max_x = W[(abs(f_signal)**2).argmax()]  # Find the x value corresponding to the maximum y value
	max_xstring = str(max_x)

	plt.title("Dominant Frequency:" + " " + max_xstring ,fontsize=20)

	plt.xscale('log')
	plt.xlim(0, 50)

	

	save("image/frequency", ext="png", close=True, verbose=True)
	print "Saved"

	'''
	CHANGE MADE HERE
	'''
	im = pyimgur.Imgur(CLIENT_ID)
	uploaded_image = im.upload_image("image/frequency.png", title="Uploaded with PyImgur")


	print "Dominant Frequency:" + max_xstring

	xlist = []
	ylist = []
	peakvalues_dict = dict()
	
	x = tdata1

	peak_values  = detect_peaks(x, mph=None, mpd=0, show=True)
	save("image/amplitude", ext="png", close=True, verbose=True)
	valley_values  = detect_peaks(x, mph=None, mpd=0, valley = True, show=True)
	save("image/valleys", ext="png", close=True, verbose=True)
	#detect_peaks(x, mph=None, mpd=0, valley=True, show=True)

	'''
	CHANGE MADE HERE
	'''
	im = pyimgur.Imgur(CLIENT_ID)
	amplitude_uploaded_image = im.upload_image("image/amplitude.png", title="Uploaded with PyImgur")


	counter =0
	bigamplitude = 0
	for p in peak_values:
		v1 = False
		v2 = False
		cp = False
		pvalley1 = p - 1
		pvalley2 = p + 1
		for v in valley_values:
			if v == pvalley1:
				firstvalley = valley_values[v]
				v1 = True
			if v == pvalley2:
				secondvalley = valley_values[v]
				v2 = True
		if v1 and v2:
			centralpeak = peak_values[p]
			cp = True

		#Now if cp is true then we evade an outlier and we have a potential cycle to test on
		if cp:
			counter = counter +1
			#generating amplitude of said potential cycle
			length1 = centralpeak - firstvalley
			length2 = centralpeak - secondvalley
			amplitude = length1 + length2
			if amplitude > 0.1:
				bigamplitude = bigamplitude + amplitude
	        
	try:
		averageamplitude = bigamplitude/counter 
	except ZeroDivisionError:
		averageamplitude = 0.16
	averageamplitude = round(averageamplitude, 5)
	averageamplitude = str(averageamplitude)+" "+"m/s^2"


	'''
	CHANGE MADE HERE WERE:
	-uploaded_image.link TO max_xstring

	-amplitude_uploaded_image.link DELETED

	REVERT BACK TO ORIGINAL AFTER TESTING
	'''
	if 3<max_x<7:
		return "Parkinson's Disease" +","+uploaded_image.link+","+averageamplitude+","+amplitude_uploaded_image.link
	elif 1<max_x<=3:
		return "Intention Tremor(MS,Stroke)" +","+uploaded_image.link+","+averageamplitude+","+amplitude_uploaded_image.link
	elif 10>max_x>=7:
		return "Essential Tremor" +","+uploaded_image.link+","+averageamplitude+","+amplitude_uploaded_image.link
	elif max_x>=10:
		return "Physiologic Tremor" +","+uploaded_image.link+","+averageamplitude+","+amplitude_uploaded_image.link
	else:
		return "No Disease" +","+uploaded_image.link+","+averageamplitude+","+amplitude_uploaded_image.link


	#plt.show()

@route('/passiveapi', method='POST')
def passive():
	upload=request.files.get('userfile')
	upload.save("postfile/", overwrite=True)
	inputarray= []
	fo=open("postfile/myTextFile.txt","r")
	for line in fo:
		inputarray.append(line.rstrip("\n"))
	fo.close()
	inputarray=list(inputarray)
	#print inputarray
	array=()
	counter=0
	#Over here we write the values from the text file into a properly formatted csv file
	data1 = []
	c = csv.writer(open("data.csv", "wb"))
	for n in inputarray:
		counter=counter+1
		if counter==1:
			t=n
		if counter==2:
			x=n
		if counter==3:
			y=n
		if counter==4:
			z=n
			array = (t,x,y,z)
			#print array[1]
			data1.append(array[1])
			c.writerow(array)
			array=()
			counter=0
	#Now we open the csv and take its values	
	data1 = []
	file = open("data.csv")
	reader = csv.reader(file)
	# print reader
	for line in reader:
		#print line[0]
		try:
			data1.append(line[1])
		except IndexError:
			continue
	# print data1
	tdata1 = np.array(data1, dtype='float')

	#Now do the image generation
	def chunks(l, n):
	    """ Yield successive n-sized chunks from l.
	    """
	    for i in xrange(0, len(l), n):
	        yield l[i:i+n]

	np.array(tdata1).tolist()
	plist = []
	for x in tdata1:
		plist.append(round(x,6))
	passivelist=list(chunks(plist, 50))

	print "Chunks created"

	counter = 0
	xlist = []
	ylist = []
	for v in passivelist:
		counter = counter +1
		x = counter
		xlist.append(x)
		py = v
		py = np.array(py, dtype='float')
		py = py - py.mean()
		W = fftfreq(py.size, d=(0.02))
		f_signal = fft(py)
		max_y = max(abs(f_signal)**2)  # Find the maximum y value
		max_x = W[(abs(f_signal)**2).argmax()]
		y = max_x 
		ylist.append(y)

	#feature 1 generation (average frequency of all values in time window)
	valuesum = 0
	valuecounter = 0
	for v in range (0,len(ylist)):
		if 0<=ylist[v]<=300:
			value = ylist[v]
			valuesum = valuesum +int(value)
			valuecounter = valuecounter +1

	f1 = valuesum/float(valuecounter)

	ncounter=0
	for v in range(0,len(ylist)):
		if 3<=ylist[v]<=6:
			ncounter = ncounter+1

	nchunks = ncounter

	#feature 2 generation (average length of parkinsonian chunks)
	runcounter = 1
	chunk = False
	chunklist = []
	for v in range(0, len(ylist)):
		if chunk==True:
			if 3<ylist[v]<6:
				chunk=True
				runcounter = runcounter+1
			else:
				chunk=False
				runamount = runcounter
				chunklist.append(runamount)
				runcounter = 1
		else:
			if 3<ylist[v]<6:
				chunk=True
			else:
				chunk=False
	if nchunks > 0:
		f2 = sum(chunklist)/float(len(chunklist))
	else:
		f2 = 0
	chunklist = []

	#Generation of feature vector
	feature_array = [f1,f2]

	print "Feature vector created"
	#DO PREDICTION OVER HERE
	#Open dump
	f = open('parkinsonclassifier.pkl', 'rb')
	Forest = pickle.load(f)
	prediction = Forest.predict(feature_array)
	prediction = str(prediction)
	f.close()

	if prediction == "['0']":
		preiction = "No Disease"
	if prediction == "['1']":
		preiction = "Low Severity"
	if prediction == "['2']":
		preiction = "Mid Severity"
	if prediction == "['3']":
		preiction = "High Severity"
	print "Prediction complete"

	a = 0
	b = 0
	c = 0

	colored = []

	dictionary = dict(zip(xlist, ylist))

	for key in dictionary:
		if 3<=dictionary[key]<=6:
			colored.append(key)



	bars = plt.bar(xlist, ylist, color='grey', edgecolor= "none")
	plt.xlabel('Time (Seconds)')
	plt.ylabel('Frequencies')
	plt.ylim([0,15])
	print "Made graph"
	
	for c in colored:
		c=c-1
		bars[c].set_facecolor('red')
		bars[c].set_edgecolor('red')

	#plt.show()

	#save("image/pinchunks", ext="png", close=True, verbose=True)
	im = pyimgur.Imgur(CLIENT_ID)
	pinchunks_uploaded_image = im.upload_image("image/pinchunks.png", title="Uploaded with PyImgur")

	return prediction+","+pinchunks_uploaded_image.link

	

	

run(host='0.0.0.0', port=8080)

# ParkinsonsBottleServerPython
