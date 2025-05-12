import pickle
from matplotlib import pyplot as plt

# loading ORL dataset
if 0:
	f = open('ORL', 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break
		
# loading CIFAR-10 dataset
if 1:
	f = open('CIFAR', 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break

# loading MNIST dataset
if 0:
	f = open('MNIST', 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix, cmap='gray')
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix, cmap='gray')
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break
	
	
	