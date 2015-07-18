RELEASE := 1
UNAME_S := $(shell uname)

LIBS = -lopencv_core -lopencv_highgui -lstasm -lopencv_objdetect -lopencv_imgproc
LIBS += -lsvm

#Linux
ifeq ($(UNAME_S), Linux)
endif

# Mac (assuming Macports is being used)
ifeq ($(UNAME_S), Darwin)
	INCDIR = -I/opt/local/include
	LIBDIR = -L/opt/local/lib
endif


# STASM
INCDIR += -I/home/febman/Desktop/stasm4.1.0/stasm
LIBDIR += -L/home/febman/Desktop/stasm4.1.0/build


# Prefer clang++ if it's installed, else use g++
CXX := $(shell which clang++)
ifndef CXX
	CXX = g++
endif

CXXFLAGS =  -std=c++11 -Wall

ifeq ($(RELEASE),1)
	CXXFLAGS +=  -O3
else
	CXXFLAGS += -O0 -g
endif

main.exe: main.o
	$(CXX) $< -o $@ $(LIBDIR) $(LIBS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@

clean:
	rm -rf *.o

clobber: clean
	rm -rf *.exe
