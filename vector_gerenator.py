import sys
import random

number_files = int(sys.argv[1]) #pega o numero de arquivos
size = int(sys.argv[2])

name_file = "file_vector_" + sys.argv[2]

k = 0

file_w = open('Input_data/' + name_file ,"w")

file_w.write(str(size)+"\n")

while k < number_files:

	i = 0

	while i < size:
		number = random.randint(-1,1)
		while number == 0:
			number = random.randint(-1,1)

		number = str(number) + ' '
		file_w.write(number)
		i = i + 1
		
	file_w.write("\n")
	file_w.close()
	k = k + 1
