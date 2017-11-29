import argparse
import matplotlib.pyplot as plt

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--trainLogFilePath', type=str, default='', help='The raw data path.')
	parser.add_argument('--testlLogFilePath', type=str, default='', help='The raw data path.')

	args = parser.parse_args()
	trainLogFilePath = args.trainLogFilePath
	testLogFilePath = args.testLogFilePath
	train_log_file_path = os.path.join(os.path.expanduser('~'),os.path.join(trainLogFilePath,'train.log'))
	test_log_file_path = os.path.join(os.path.expanduser('~'),os.path.join(testLogFilePath,'test.log'))

	train_loss_list = []
	with open(train_log_file_path) as f:
    	next(f)
    	for line in f:
    		train_loss_list.append(line)

    test_loss_list = []
	with open(test_log_file_path) as f:
    	next(f)
    	for line in f:
    		test_loss_list.append(line)

    plt.figure(1)
    plt.plot([train_loss_list, 'ro')
	plt.axis([-1, 1, 0, len(train_loss_list)])
	plt.show()

	plt.figure(2)
	plt.plot([test_loss_list, 'ro')
	plt.axis([-1, 1, 0, len(test_loss_list)])
	plt.show()


if __name__ == '__main__':
	main()