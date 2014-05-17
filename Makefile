CC = g++

CFLAGS = -Wall -pg

CFLAGS += -msse2

RM = rm -f

OBJS =  main.o matrix.o vector.o pls_sequential.o

MAIN = pls

$(MAIN): $(OBJS)
	@echo ""
	@echo " --- COMPILANDO PROGRAMA ---"
	@$(CC) $(CFLAGS) $(OBJS) -lnsl -lm -o $(MAIN)
	@echo ""

%.o: %.c %.h
	@echo " --- COMPILANDO OBJETO \"$@\""
	@$(CC) $(CFLAGS) $< -c 

clean:
	$(RM) $(MAIN) *.o
	clear

