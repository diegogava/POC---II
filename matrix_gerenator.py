import sys
import random

number_files = int(sys.argv[1]) #pega o numero de arquivos (isso eh passado por argumento)
lines = int(sys.argv[2])
columns = int(sys.argv[3])

name_file = "file_matrix_" + sys.argv[2]

k = 0

file_w = open('Input_data/'+ name_file ,"w")

file_w.write(str(lines) + " "+str(columns)+"\n")

while k < number_files:

	i = 0

	while i < lines:
		j = 0
		while j < columns:
			number = random.random()
			number = str(number) + ' '
			file_w.write(number)
			j = j + 1
		i = i + 1
		file_w.write("\n")

	file_w.close()
	k = k + 1
