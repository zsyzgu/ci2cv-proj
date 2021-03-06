INCLUDES =  -I/usr/local/include/opencv \
	-I/Users/zsyzgu/编程/ARCooperation/face-analysis-sdk-master/src \
	-I/Users/zsyzgu/编程/ARCooperation/face-analysis-sdk-master/build/src

LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_objdetect \
	-lutilities -lclmTracker -lavatarAnim

LIBDIRS = -L/usr/local/lib \
	-L/Users/zsyzgu/编程/ARCooperation/face-analysis-sdk-master/build/lib

OPT = -O3 -Wno-deprecated

CC = g++

#.PHONY: all clean

OBJS = 

all: main

clean:
	rm -f *.o *~

%.o: %.cpp
	$(CC) -c $(INCLUDES) $+ $(OPT)

main: main.o client.o frame.o
	$(CC) $(LIBDIRS) $(LIBS) -o $@ $+ $(OPT)
