import sys
import getopt
from collections import Counter
import logging as log
import numpy

#parse the input file returning the data used to train and test the perceptron algorithm
def getDataFromFile(fileName):
	f = open(fileName)
	data = []
	c = Counter()
	i = 1.0
	classification = []
	#create the vocabulary
	for line in f:
		c = c + Counter(line.split()[1:])
		log.info(str(i*100.0/4000.0))
		i = i + 1
	
	f.close()
	#remove all records from vocabulary with less then 30 counts
	for item in c.keys():
		if c[item] < 30:
			log.debug('Removed item: [%s]', item)
			del c[item]

	data = convertText2FeatureArray(fileName, c)
	
	return data, c

#transform the mail body to a vector of 0's and 1's. Also, extracts the classification of each email
def convertText2FeatureArray(fileName, vocabulary):
	data = []
        f = open(fileName)
        for line in f:
                lineSplit = line.split()
                words = lineSplit[1:]
                emailFeatures = []
                for key in vocabulary.keys():
                        if key in words:
                                emailFeatures.append(1)
                        else:
                                emailFeatures.append(0)
                data.append([int(lineSplit[0]), numpy.array(emailFeatures)])

        f.close()
        log.debug(vocabulary)

	return data


#Trains a perceptron classifier using the examples provided to the function, and return the final classification vector, the number of updates (mistakes) performed, and the number of passes through the data, respectively. For the corner case of w.x = 0, predict the +1 (spam) class.
def perceptron_train(data):
	allCorrect = False
	dimension = len(data[0][1])
	w = numpy.array([0] * dimension)
	errorCount = 0
	step = 0

	while not allCorrect:
		step = step + 1
		allCorrect = True
		for record in data:
			r = numpy.dot(w, record[1])
			log.debug(w)
			if r > 0 and record[0] == 0:
				#error: missclassified a non spam email as spam
				errorCount += 1
				allCorrect = False
				w = w + record[1] * -1
			elif r <= 0 and record[0] == 1:
				#error: missclassified a spam email as non spam
				errorCount += 1
				allCorrect = False
				w = w + record[1] * 1
	
	return w, errorCount, step

def perceptron_test(data, w):
        dimension = len(data[0][1])
        w = numpy.array([0] * dimension)
        errorCount = 0.0
	total = 0.0

        for record in data:
		total += 1.0
        	r = numpy.dot(w, record[1])
                log.debug(w)
                if r > 0 and record[0] == 0:
                	#error: missclassified a non spam email as spam
                        errorCount += 1.0
		elif r <= 0 and record[0] == 1:
                	#error: missclassified a spam email as non spam
                        errorCount += 1.0


        return errorCount/total

def main(argv=sys.argv):
	log.basicConfig(format='%(levelname)s:%(message)s', filename='/tmp/run.log', level=log.INFO)		
	log.info('########### Execution starts ###########')	
	try:
        	opts, args = getopt.getopt(sys.argv[1:], 'v:t:')
    	except getopt.GetoptError as err:
	        print str(err) 
	        sys.exit(2)
	trainingFileName = None
    	validationFileName = None

    	for o, a in opts:
        	if o == '-t':
			log.debug('Configuring testing file name: %s', a)
         	   	trainingFileName = a
        	elif o == '-v':
			log.debug('Configuring validation file name: %s', a)
            		validationFileName = a
        	else:
	            assert False, "unhandled option"

	if trainingFileName != None:
		data, vocabulary = getDataFromFile(trainingFileName)
		w, errorCount, step = perceptron_train(data)
		print 'Updates count:' + str(errorCount)
		print 'Steps run:' + str(step)

		if validationFileName != None:
			data = convertText2FeatureArray(validationFileName, vocabulary)
			errorRate = perceptron_test(data, w)
			
			print("%.2f" % errorRate)

	log.info('########### Execution finished ###########')

if __name__ == "__main__":
    sys.exit(main())
