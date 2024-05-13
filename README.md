# Parallel Background and Foregroud Mask Calculation using multiprocessing MPI / multithreading OpenMP

It’s a technique for removing static background by subtracting a set of images to obtain a final mask with the foreground objects. This is a basic technique which will work only on tiny/small motion changes in the images that are given.

The input consists of:
- M RGB (3 channels) images.
- 1 test image from which the foreground object will be masked.
- Threshold T.

The output consists of:
- 1 RGB image of the background.
- 1 Binary Mask highlighting the foreground object in the test image.

## The Dataset:
A sequence of 500 images taken using a fixed camera. The sequence starts with a clear street view then a
man crosses the street and later on a car passes by too.

## Performance Comparison:

The program runs all 3 types of processing in every run: Serial, Multithreading using openMP, and
Multiprocessing using MPI. The time is calculated for each part and printed.

Some observations include:

- Serial Execution ALWAYS have larger execution time, slower processing.
- In the case of only 2 processes/ threads, no noticeable speedup takes place.
- In the beginning (2,3): MPI multiprocessing had a larger execution time than openMP multithreading
because threads are lighter (less overhead) than the processes.
- Afterwards (4,5), MPI multiprocessing had smaller execution time than openMP multithreading
because the parallelism speedup effect of the MPI processes covered for the overhead.
- Lastly, both MPI multiprocessing and openMP multithreading started to take longer execution times
due to increased communication overhead in comparison with parallelization gains.
- Best performance for both methods took place at 5 threads for openMP and 5 processes for MPI.
- Serial processing is ALWAYS faster than MPI multiprocessing in the foreground subtraction part as
the communication overhead is larger than the parallelism gains.
- As the number of threads exceeds 5, Serial processing becomes faster than openMP multithreading
as well as in the foreground subtraction part as the communication overhead becomes larger than the
parallelism gains.

## Execution notes:
To compile on macos: 
```bash
/opt/homebrew/opt/llvm/bin/clang++ -Xpreprocessor -fopenmp -I/opt/homebrew/opt/llvm/include -I/opt/homebrew/Cellar/open-mpi/5.0.3/include -L/opt/homebrew/opt/llvm/lib -L/opt/homebrew/Cellar/open-mpi/5.0.3/lib -lomp -lmpi Project_Code.cpp -o project
```

To run: (number specifies parallel processes count for MPI and parallel threads count for OpenMP)
```bash
mpirun -np 5 ./BackgroundSubtraction
```