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
		classification = -1 if lineSplit[0] == '0' else 1
                data.append([classification, numpy.array(emailFeatures)])

        f.close()
        log.debug(vocabulary)

	return data


def pegasos_svm_train(data, lambd):
	max_epoch = 20
	current_epoch = 0
	step = 0
	dimension = len(data[0][1])
	w = numpy.array([0] * dimension)
	f_w = 0.0
	while current_epoch < max_epoch:
		step += 1.0
		n_t  = 1.0/(step * lambd)
		total = 0.0
		m = 1.0
		errorCount = 0.0

		for record in data:
			r = numpy.dot(w, record[1])
			total += 1.0 - (record[0] * numpy.dot(w, record[1]))
			m += 1.0
			
			if r * record[0] < 0.0:
				errorCount += 1.0	
			
			if r * record[0] < 1.0:
                	        w = (1.0 - (1.0/step)) * w + ((n_t * record[0]) * record[1])
			else:
        	                w = (1.0 - (1.0/step)) * w

		#calculate f(w)
		f_w = (lambd/2.0) * pow(numpy.linalg.norm(w),2)
		f_w = f_w + (total/m)

		current_epoch += 1
		#print 'Epoch: {0} f(w): {1}'.format(current_epoch, f_w)
		#print 'Epoch: {0} error: {1}'.format(current_epoch, errorCount/m)

	print 'Train Error rate: {0}'.format(errorCount/m)
	print 'Train f(w)', f_w
	return w

	
def pegasos_svm_test(data, w, lambd):
        errorCount = 0.0
	total = 0.0
	m = 1.0

	for record in data:
		r = numpy.dot(w, record[1])
                total += 1.0 - (record[0] * numpy.dot(w, record[1]))
                m += 1.0

                if r * record[0] < 0:
			errorCount += 1

                #calculate f(w)
                f_w = (lambd/2.0) * pow(numpy.linalg.norm(w),2)
                f_w = f_w + (total/m)


        return errorCount/m, f_w


def main(argv=sys.argv):
	log.basicConfig(format='%(levelname)s:%(message)s', filename='/tmp/run.log', level=log.DEBUG)		
	log.info('########### Execution starts ###########')	
	try:
        	opts, args = getopt.getopt(sys.argv[1:], 'v:t:e:')
    	except getopt.GetoptError as err:
	        print str(err) 
	        sys.exit(2)
	trainingFileName = None
    	validationFileName = None
	exponent = 5
    	for o, a in opts:
        	if o == '-t':
			log.debug('Configuring testing file name: %s', a)
         	   	trainingFileName = a
        	elif o == '-v':
			log.debug('Configuring validation file name: %s', a)
            		validationFileName = a
		elif o == '-e':
			exponent = int(a)
        	else:
	            assert False, "unhandled option"

	print 'lambd = 2^', exponent

	if trainingFileName != None:
		data, vocabulary = getDataFromFile(trainingFileName)
		w = pegasos_svm_train(data, pow(2,exponent))

		if validationFileName != None:
			data = convertText2FeatureArray(validationFileName, vocabulary)
			errorRate, f_w = pegasos_svm_test(data, w, pow(2,exponent))
			print('Validation %.6f' % f_w),
			print('Validation Error Rate %.6f' % errorRate)

	log.info('########### Execution finished ###########')

if __name__ == "__main__":
    sys.exit(main())
