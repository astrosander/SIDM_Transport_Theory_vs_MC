#--global
COMPILE=gfortran
hdf5dirlib=/usr/lib/x86_64-linux-gnu/hdf5/serial/
hdf5dirinc=/usr/include/hdf5/serial/
performance_analyze=0
use_hdf5=1
maximum_compatible=1
FC= mpif90
#FC= mpiifort
libdir:=../source/
ifeq ($(COMPILE), ifort)
	FLAG=  -O3 -traceback  \
	      -L$(libdir) -I$(libdir) -lgnc \
	      -check bounds -check uninit -check pointers \
		   -traceback -debug -fpp 
	#FLAG+= -f90=ifort 

	ifeq ($(maximum_compatible), 1)
		FLAG+=-align none
	endif
    ifeq ($(use_hdf5),1)
    	FLAG+= -DHDF5
    	FLAG+= -I$(hdf5dirinc) -L$(hdf5dirlib) -lhdf5_fortran -lhdf5hl_fortran -lhdf5 -lhdf5_hl
    endif

    ifeq ($(performance_analyze), 1)
                FLAG+= -g -DNDEBUG
    endif

else
	FLAG=-I$(libdir) -L$(libdir)  -lgnc -fstack-arrays \
	 -O3 -fcheck=all -fbacktrace -cpp   -fimplicit-none  -malign-double
    ifeq ($(use_hdf5),1)
    	FLAG+= -DHDF5
    	FLAG+= -I$(hdf5dirinc) -L$(hdf5dirlib) -lhdf5_fortran -lhdf5hl_fortran -lhdf5 -lhdf5_hl  
    endif
        ifeq ($(performance_analyze), 1)
		FLAG+=-pg -g
	endif
    ifeq ($(performance_analyze), 2)
		FLAG+=-fprofile-arcs -ftest-coverage
	endif
endif

all: main ini pro cfuns
	 

% : %.f90
	$(FC) $< -o $@ $(FLAG)
	execstack -s $@

.PHONY:clean
clean:
	rm *.gcda
	rm *.gcno
